"""Generate the editable Cobalt exterior and its review cameras in Blender."""
import argparse
from bisect import bisect_right
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt


ROOT = Path(__file__).resolve().parent
NOSE, TAIL = -2.2395, 2.2395
AXLES = (-1.360, 1.408)
HUB_Z, TIRE_R = 0.326, 0.326


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def lerp(a, b, t):
    return a + (b - a) * t


def fair_curve(stations):
    """Natural cubic interpolant for slowly changing highlights on exterior panels."""
    xs,ys=zip(*stations)
    intervals=[b-a for a,b in zip(xs,xs[1:])]
    n=len(xs)
    upper=[0.0]*n
    rhs=[0.0]*n
    for i in range(1,n-1):
        a,b=intervals[i-1],intervals[i]
        pivot=2*(a+b)-a*upper[i-1]
        upper[i]=b/pivot
        rhs[i]=(6*((ys[i+1]-ys[i])/b-(ys[i]-ys[i-1])/a)-a*rhs[i-1])/pivot
    for i in range(n-2,0,-1):
        rhs[i]-=upper[i]*rhs[i+1]
    def sample(x):
        x=max(xs[0],min(xs[-1],x))
        k=min(n-2,bisect_right(xs,x)-1)
        h=intervals[k]
        b=(x-xs[k])/h
        a=1-b
        return a*ys[k]+b*ys[k+1]+((a*a*a-a)*rhs[k]+(b*b*b-b)*rhs[k+1])*h*h/6
    return sample


def smooth(points, x):
    """Shape-preserving cubic interpolation of measured guide stations."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(points, points[1:])]
    tangents = [slopes[0]]
    for a, b in zip(slopes, slopes[1:]):
        tangents.append(2 * a * b / (a + b) if a * b > 0 else 0.0)
    tangents.append(slopes[-1])
    for i, ((left, y0), (right, y1)) in enumerate(zip(points, points[1:])):
        if x <= right:
            d = right - left
            p = (x - left) / d
            return (lerp(y0, y1, p*p*(3-2*p)) + d * p * (1-p) *
                    ((1-p) * tangents[i] - p * tangents[i+1]))


def material(name, rgb, roughness=.35, metallic=0, coat=0, emission=0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = .18
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*rgb, 1)
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def micro_bump(mat, scale, strength, distance):
    nodes = mat.node_tree.nodes
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    coordinate = nodes.new("ShaderNodeTexCoord")
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    links = mat.node_tree.links
    links.new(coordinate.outputs["Object"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], nodes.get("Principled BSDF").inputs["Normal"])


def collection(name, parent=None):
    out = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(out)
    return out


def move(obj, coll):
    for prior in list(obj.users_collection):
        prior.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


def mesh(name, vertices, faces, mat, coll, shaded=True, normal=None):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.validate()
    data.update()
    if normal and sum(p.normal.dot(Vector(normal)) * p.area for p in data.polygons) < 0:
        for p in data.polygons:
            p.flip()
        data.update()
    obj = bpy.data.objects.new(name, data)
    coll.objects.link(obj)
    obj.data.materials.append(mat)
    for p in data.polygons:
        p.use_smooth = shaded
    return obj


def patch(name, fn, rows, cols, mat, coll, normal=None, thickness=0):
    vertices = [fn(i/rows, j/cols) for i in range(rows+1) for j in range(cols+1)]
    faces = [(i*(cols+1)+j, (i+1)*(cols+1)+j, (i+1)*(cols+1)+j+1, i*(cols+1)+j+1)
             for i in range(rows) for j in range(cols)]
    obj = mesh(name, vertices, faces, mat, coll, normal=normal)
    uv = obj.data.uv_layers.new(name="Surface")
    for loop in obj.data.loops:
        i, j = divmod(loop.vertex_index, cols+1)
        uv.data[loop.index].uv = (i/rows, j/cols)
    if thickness:
        mod = obj.modifiers.new("Sheet thickness", "SOLIDIFY")
        mod.thickness = thickness
        mod.offset = -1
    return obj


def path(name, coordinates, radius, mat, coll, closed=False):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 12
    data.bevel_depth = radius
    data.bevel_resolution = 3
    data.use_fill_caps = True
    spline = data.splines.new("POLY")
    spline.points.add(len(coordinates)-1)
    for p, v in zip(spline.points, coordinates):
        p.co = (*v, 1)
    spline.use_cyclic_u = closed
    obj = bpy.data.objects.new(name, data)
    coll.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def perimeter(points, count=5):
    """Rounded closed silhouette through authored corner/intermediate guides."""
    result = []
    for i, b in enumerate(points):
        a, c, d = points[i-1], points[(i+1) % len(points)], points[(i+2) % len(points)]
        for k in range(count):
            t = k / count
            result.append(tuple(lerp(bb, cc, t*t*(3-2*t)) + .30 * t*(1-t) *
                                ((1-t)*(cc-aa) - t*(dd-bb))
                                for aa, bb, cc, dd in zip(a,b,c,d)))
    return result


def open_contour(points, step=.013):
    """Sample a fair open feature line before projecting it onto the panel."""
    distances=[0.]
    for a,b in zip(points,points[1:]):
        distances.append(distances[-1]+math.dist(a,b))
    xs=fair_curve(list(zip(distances,[p[0] for p in points])))
    ys=fair_curve(list(zip(distances,[p[1] for p in points])))
    steps=max(2,math.ceil(distances[-1]/step))
    return [(xs(t),ys(t)) for t in (distances[-1]*i/steps for i in range(steps+1))]


def island(name, points, mapping, mat, coll, normal=None, crown=0):
    pts = perimeter(points, 9)
    center = tuple(sum(p[i] for p in pts)/len(pts) for i in range(2))
    vertices = [mapping(*center, crown)]
    rings = [i/20 for i in range(1,21)]
    for r in rings:
        vertices.extend(mapping(*(lerp(q, p, r) for p, q in zip(pt, center)), crown*(1-r*r))
                        for pt in pts)
    n = len(pts)
    faces = [(0, 1+(i+1)%n, 1+i) for i in range(n)]
    for k in range(len(rings)-1):
        start = 1+k*n
        faces.extend((start+i, start+(i+1)%n, start+n+(i+1)%n, start+n+i) for i in range(n))
    return mesh(name, vertices, faces, mat, coll, normal=normal), pts


def inset(points, ratio):
    c = tuple(sum(p[i] for p in points)/len(points) for i in range(2))
    return [tuple(lerp(q, p, ratio) for p,q in zip(pt,c)) for pt in points]


def ellipsoid(name, where, size, mat, coll):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=where)
    obj = move(bpy.context.object, coll)
    obj.name = name
    obj.scale = size
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def block(name, where, size, bevel, mat, coll):
    bpy.ops.mesh.primitive_cube_add(size=1, location=where)
    obj = move(bpy.context.object, coll)
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mod = obj.modifiers.new("Rounded edges", "BEVEL")
    mod.width, mod.segments = bevel, 3
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    obj.data.materials.append(mat)
    return obj


def lathe(name, origin, profile, mat, coll, axis=1, n=72):
    vertices = []
    for axial, radius in profile:
        for i in range(n):
            t = math.tau * i / n
            p = (math.cos(t)*radius, axial, math.sin(t)*radius)
            if axis == 0:
                p = (axial, p[0], p[2])
            vertices.append(tuple(a+b for a,b in zip(origin,p)))
    faces = [(r*n+i, r*n+(i+1)%n, ((r+1)%len(profile))*n+(i+1)%n,
              ((r+1)%len(profile))*n+i) for r in range(len(profile)) for i in range(n)]
    obj = mesh(name, vertices, faces, mat, coll)
    # Locating wheel origins at the hubs makes each assembly easy to spin.
    for v in obj.data.vertices:
        v.co -= Vector(origin)
    obj.location = origin
    return obj


section_width=fair_curve([(NOSE,.873),(-1.52,.865),(-.60,.851),(.32,.851),
                          (1.34,.868),(1.87,.860),(TAIL,.847)])
BELT = [(NOSE,.922),(-2.10,.927),(-1.82,.948),(-1.42,.979),(-1.02,1.017),
        (-.82,1.009),(-.48,1.021),(-.12,1.037),(.39,1.052),(1.16,1.059),(1.60,1.082),(1.78,1.092),
        (1.95,1.078),(2.06,1.060),(2.17,1.042),(TAIL,1.020)]
CROWN = [(NOSE,-.103),(-1.89,-.046),(-1.42,.007),(-.80,.018),(.70,.020),
         (1.40,.044),(1.76,.056),(2.03,.061),(TAIL,.064)]
SILL = [(NOSE,.191),(-2.12,.190),(-1.68,.199),(-.90,.208),(.80,.209),
        (1.85,.246),(2.04,.271),(2.15,.285),(TAIL,.257)]
side_section=fair_curve([(0,.029),(.13,.021),(.34,.013),(.58,.006),
                         (.76,.009),(.86,.024),(.94,.053),(1,.092)])
front_section=fair_curve([(.19,.781),(.27,.818),(.45,.842),(.60,.851),
                          (.76,.842),(.855,.829),(.922,.815)])

# Side-view crown above the lamp, sampled in metres from the original elevation.
bonnet_profile=fair_curve([(-2.10,.819),(-2.049,.853),(-1.980,.891),
                           (-1.888,.920),(-1.795,.944),(-1.680,.973),
                           (-1.496,1.006),(-1.311,1.030),(-1.223,1.040),
                           (-1.140,1.044),(-1.00,1.033)])


def arch_curve(x):
    z = smooth(SILL, x)
    for center in AXLES:
        dx = abs(x-center)
        if dx < .387:
            weight = math.sqrt(max(0, 1-(dx/.387)**2))
            z = max(z, .214 + .480*weight)
    return z


def flank_y(x, z):
    belt = smooth(BELT,x)
    t = (z-smooth(SILL,x))/(belt-smooth(SILL,x))
    inset=side_section(t)
    if t>1:
        inset+=(t-1)*(side_section(1)-side_section(.998))/.002
    flare=sum(math.exp(-((x-c)/.43)**2) for c in AXLES)
    shoulder=belt-.126-.033*math.exp(-((x+.11)/.85)**2)
    aft=min(1,max(0,(x-.29)/.67))
    aft=aft*aft*(3-2*aft)
    shoulder=lerp(shoulder,.943+.006*min(x,2),aft)
    swage=lerp(.0055,.009,aft)*math.tanh((z-shoulder)/.027)*math.exp(-((z-shoulder)/.13)**2)
    # Shallow pressed line under the door handles climbs towards the rear wheel.
    door_weight=math.exp(-((x+.05)/1.20)**6)
    door_line=.507+.021*x
    character=.015*door_weight*math.tanh((z-door_line)/.011)*math.exp(-((z-door_line)/.073)**2)
    rocker_height=.353+.015*x
    rocker=.009*math.tanh((z-rocker_height)/.017)*math.exp(-((z-rocker_height)/.047)**2)
    # Broad arch crowns carry the measured track-side width without moving the lamps.
    shape=section_width(x)-inset+swage+rocker+character+flare*(.0155+.010*math.exp(-((z-.68)/.17)**2))
    blend=max(0,1-(x-NOSE)/1.18)**3
    # Round the rear corner in plan: the side tucks in with a quarter-ellipse so it
    # arrives tangent to the wrapped tail panel instead of meeting it at ~45 degrees.
    u=min(1,max(0,(x-(TAIL-TAIL_TUCK_LENGTH))/TAIL_TUCK_LENGTH))
    tuck=TAIL_TUCK*(1-math.sqrt(max(0,1-u*u)))
    return lerp(shape,front_section(z),blend)-tuck


TAIL_TUCK, TAIL_TUCK_LENGTH = .054, .245


def skin_side(x, z, side, offset=0):
    y=side*flank_y(x,z)
    return (shell_x(x,y,z),y+side*offset,z)


# Columns are side-view row, the two transverse coordinates on the opening
# border, and the longitudinal correction at each of those borders in pixels.
LAMP_PROFILE = {
    -1: [(213,.756,.760,1,1),(217,.733,.773,3,-2),(221,.701,.782,7,-3),
         (225,.655,.789,11,-4),(229,.603,.793,12,-5),(233,.573,.796,10,-6),
         (238,.552,.798,7,-7),(243,.540,.800,4,-8),(250,.524,.800,2,-10),
         (255,.516,.801,1,-11),(260,.505,.796,2,-10),(265,.493,.767,3,-7),
         (269,.495,.650,5,-3),(273,.506,.620,7,0)],
    1: [(189,.728,.682,3,-5),(193,.763,.651,-4,-1),(196,.781,.645,-3,-3),
        (200,.796,.640,-2,-4),(207,.802,.630,-1,-6),(214,.803,.619,-1,-6),
        (220,.804,.607,0,-5),(227,.804,.594,1,-2),(231,.804,.588,2,0),
        (235,.804,.581,5,1),(238,.802,.575,7,2),(241,.792,.568,6,3),
        (245,.592,.560,3,2)],
}
LAMP_PROFILE_CURVES = {
    sign: [fair_curve([(station[0], station[column]) for station in stations])
           for column in range(1,5)]
    for sign,stations in LAMP_PROFILE.items()
}


def fit_lamp_profile(sign, y, z):
    # Register both lenses to the same side-view datum used by the comparison.
    row=400-z*.9973076*315/1.505
    transverse=abs(y)*.9783335
    if sign<0:
        target_row=.89242*row+24.182
        y_bounds=(.491,.807)
        row_bounds=(213,273)
    else:
        target_row=.87238*row+26.572
        y_bounds=(.553,.808)
        row_bounds=(189,245)
    first,last=row_bounds
    y_min,y_max=y_bounds
    fade=min(1,max(0,(transverse-y_min+.080)/.080),
             max(0,(y_max+.089-transverse)/.089),
             max(0,(target_row-first+24)/24),max(0,(last+30-target_row)/30))
    if not fade:
        return 0,0
    fade=fade*fade*(3-2*fade)
    near,far,near_dx,far_dx=LAMP_PROFILE_CURVES[sign]
    position=max(first,min(last,target_row))
    a,b=near(position),far(position)
    # The two rails meet at each lens tip. Keep that interpolation continuous.
    fraction=min(1,max(0,.5+(transverse-(a+b)*.5)*(b-a)/max(.012**2,(b-a)**2)))
    cross_weight=fraction*fraction*(3-2*fraction)
    dx=lerp(near_dx(position),far_dx(position),cross_weight)*4.479/965/1.0053531
    dz=(row-target_row)*1.505/315/.9973076
    end=NOSE if sign<0 else TAIL
    shoulder=smooth(BELT,end)
    edge=flank_y(end,shoulder)
    cross=min(1,abs(y)/edge)
    upper=nose_height(cross) if sign<0 else shoulder+smooth(CROWN,end)*(1-cross*cross)
    upper_weight=min(1,max(0,(upper-z)/.033))
    lateral_weight=min(1,max(0,(flank_y(end,z)-abs(y))/.030))
    # The adjoining side and deck charts retain their common perimeter.
    # Vertical corrections fade to zero at those joins.
    upper_weight=upper_weight*upper_weight*(3-2*upper_weight)
    lateral_weight=lateral_weight*lateral_weight*(3-2*lateral_weight)
    return fade*upper_weight*dx,fade*upper_weight*lateral_weight*dz


def fascia(sign, y, z, outset=0):
    if sign == -1:
        # Map the measured frontal drawing onto the round plan-view bumper.
        f = min(1.13, abs(y)/.844)
        center = smooth([(.18,.064),(.24,.034),(.38,.003),(.46,.001),(.52,.015),
                         (.56,.022),(.65,.048),(.77,.104),(.82,.133),(.93,.175)],z)
        wrap = smooth([(.18,.17),(.39,.172),(.54,.190),(.70,.290),(.82,.394),(.925,.505)],z)
        zone=math.exp(-((f-.80)/.22)**2)
        relief=zone*(.010*math.exp(-((z-.431)/.030)**2)-.013*math.exp(-((z-.475)/.035)**2)
                     -.012*math.exp(-((z-.278)/.027)**2))
        rib_y=smooth([(.235,.306),(.29,.335),(.37,.372),(.47,.414),(.55,.459),(.61,.487)],z)
        ramp=max(0,min(1,(z-.228)/.056,(.615-z)/.064))
        ramp=ramp*ramp*(3-2*ramp)
        dy=abs(y)-rib_y
        relief+=ramp*.012*math.tanh(dy/.024)*math.exp(-(dy/.075)**2)
        q=abs(y)/.882
        plan=smooth([(0,0),(.32,.011),(.50,.046),(.63,.096),(.74,.158),(.82,.222),
                     (.89,.312),(.94,.426),(.98,.535),(1,.599),(1.07,.674)],q)
        shoulder=wrap*(.16*f*f+.84*f**4.6)
        # Enforce the minimum setback observed in plan at this transverse station.
        # Soft maximum; a 4 mm blend radius left a crumpled crease, 18 mm reads as a fillet.
        arc=.5*(plan+shoulder+math.sqrt((plan-shoulder)**2+.018**2))
        # The upper lens tapers farther back along the fender in the side view.
        # This field also positions the matching bumper opening and hood landing.
        outer_lamp=math.exp(-((abs(y)-.778)/.101)**2)
        upper_lamp=.130*math.exp(-((z-.911)/.112)**2)*outer_lamp
        lower_lamp=.039*math.exp(-((z-.615)/.122)**2-((abs(y)-.500)/.138)**2)
        lens_lift=.025*math.exp(-((z-.584)/.066)**2-((abs(y)-.632)/.137)**2)
        depth_fit,height_fit=fit_lamp_profile(sign,y,z+lens_lift)
        return NOSE + center + arc+relief+upper_lamp+lower_lamp+depth_fit-outset,y,z+lens_lift+height_fit
    if sign == 1:
        f=min(1.15,abs(y)/.833)
        rake=smooth([(.20,.083),(.27,.029),(.36,.007),(.45,.003),(.56,.022),
                     (.629,.031),(.65,.049),(.81,.043),(.96,.068),(1.085,.095)],z)
        turn=smooth([(.20,.208),(.50,.236),(.68,.254),(.87,.285),(1.08,.354)],z)
        corner=.058*math.exp(-((z-.953)/.167)**2-((abs(y)-.797)/.098)**2)
        inner_return=.095*math.exp(-((z-.958)/.174)**2-((abs(y)-.65)/.095)**2)
        # In profile the lower outer lens tapers toward the tail, following the
        # shut line instead of keeping the upper shoulder's setback.
        lower_return=.112*math.exp(-((z-.734)/.146)**2-((abs(y)-.789)/.062)**2)
        depth_fit,height_fit=fit_lamp_profile(sign,y,z)
        lower_roll=.014*math.exp(-((z-.344)/.029)**2-((abs(y)-.618)/.205)**2)
        return (TAIL-rake-turn*(.23*f*f+.77*f**5)+corner+inner_return+lower_return+lower_roll+depth_fit+outset,
                y,z+height_fit)


def shell_x(x, y, z):
    def influence(distance,reach):
        t=max(0,1-distance/reach)
        return t*t*(3-2*t)
    if x<NOSE+1.40:
        return x+influence(x-NOSE,1.40)*(fascia(-1,y,z)[0]-NOSE)
    if x>TAIL-1.10:
        return x+influence(TAIL-x,1.10)*(fascia(1,y,z)[0]-TAIL)
    return x


def nose_height(f):
    return smooth([(0,.819),(.54,.820),(.72,.821),(.82,.845),(.91,.883),
                   (.957,.912),(1,.922)],abs(f))


PANEL_COLS = 72


def relax_grid(grid, iterations=80, weight=.6):
    """Laplace-relax interior (station, height) nodes with the outline held fixed."""
    g=[[list(p) for p in row] for row in grid]
    rows,cols=len(g)-1,len(g[0])-1
    for _ in range(iterations):
        for i in range(1,rows):
            above,row,below=g[i-1],g[i],g[i+1]
            for j in range(1,cols):
                p=row[j]
                for k in (0,1):
                    avg=.25*(above[j][k]+below[j][k]+row[j-1][k]+row[j+1][k])
                    p[k]+=weight*(avg-p[k])
    return [[tuple(p) for p in row] for row in g]


def surface_normals(obj, grid, surface, direction, step=.002):
    """Shade a patch from its design surface, not from its facets."""
    axis=Vector(direction)
    normals=[]
    for row in grid:
        for q,z in row:
            dq=Vector(surface(q+step,z))-Vector(surface(q-step,z))
            dz=Vector(surface(q,z+step))-Vector(surface(q,z-step))
            n=dq.cross(dz).normalized()
            if n.dot(axis)<0:
                n.negate()
            normals.append(n)
    obj.data.normals_split_custom_set_from_vertices(normals)


def exterior():
    cuts = [
        lambda v:NOSE,
        lambda v:smooth([(0,-.875),(.52,-.893),(.84,-.897),(1,-.861)],v),
        lambda v:lerp(.217,.252,v),
        lambda v:smooth([(0,.812),(.37,.948),(.67,1.105),(1,1.276)],v),
        lambda v:TAIL,
    ]
    def station_at(world_x,z):
        lo,hi=NOSE,TAIL
        for _ in range(28):
            mid=(lo+hi)/2
            if skin_side(mid,z,1)[0]<world_x:
                lo=mid
            else:
                hi=mid
        return (lo+hi)/2

    def arch_border(axle):
        result=[]
        for k in range(81):
            angle=math.pi*k/80
            x=axle-.387*math.cos(angle)
            z=.220+.465*math.sin(angle)
            result.append((station_at(x,z),z))
        return result

    arches=[arch_border(c) for c in AXLES]
    for s, suffix in [(-1,"L"),(1,"R")]:
        for i,name in enumerate(("Front fender","Front door","Rear door","Rear quarter")):
            gap=.0020
            def edges(v):
                return cuts[i](v)+(gap if i else 0),cuts[i+1](v)-(gap if i<3 else 0)
            left,right=edges(0)
            border=[]
            def sill_run(a,b):
                sections=max(2,math.ceil((b-a)/.028))
                return [(q,smooth(SILL,q)) for q in (lerp(a,b,k/sections) for k in range(sections+1))]
            if i in (0,3):
                arc=arches[0 if i==0 else 1]
                border=sill_run(left,arc[0][0])+arc+sill_run(arc[-1][0],right)
            else:
                border=sill_run(left,right)
            lengths=[0.]
            for (qa,za),(qb,zb) in zip(border,border[1:]):
                lengths.append(lengths[-1]+math.hypot(qb-qa,.24*(zb-za)))
            # Independent panels sample one side surface. Stations follow the actual
            # wheel cut; the upper rows follow the door breaks and the belt edge.
            rows,cols=len(border)-1,PANEL_COLS
            def seed(index,v):
                q0,z0=border[index]
                t=lengths[index]/lengths[-1]
                a,b=edges(v)
                q=lerp(a,b,t)+(1-v)*(q0-lerp(left,right,t))
                return q,lerp(z0,smooth(BELT,q),v)
            # The arch makes neighbouring seed columns nearly coincide; relax the
            # (station, height) grid so no sliver quads crease the paint.
            grid=relax_grid([[seed(i,j/cols) for j in range(cols+1)] for i in range(rows+1)])
            def panel(u,v,grid=grid):
                return skin_side(*grid[round(u*rows)][round(v*cols)],s)
            if i in (0,3):
                # Fill the concave arch outline in station-height coordinates. A loft
                # across this concavity can double back into the wheel opening.
                outline=border+[grid[-1][j] for j in range(1,cols+1)]
                outline += [grid[k][-1] for k in range(rows-1,-1,-1)]
                outline += [grid[0][j] for j in range(cols-1,0,-1)]
                physical=[(skin_side(q,z,s)[0],z) for q,z in outline]
                stations={}
                def wing(x,z):
                    q=station_at(x,z)
                    stations[x,z]=q
                    return (x,s*flank_y(q,z),z)
                def wing_normal(x,z):
                    q=stations[x,z]
                    h=.0005
                    tangent=Vector(skin_side(q+h,z,s))-Vector(skin_side(q-h,z,s))
                    rise=Vector(skin_side(q,z+h,s))-Vector(skin_side(q,z-h,s))
                    return tangent.cross(rise).normalized()
                end_sign=-1 if i==0 else 1
                cut=side_bumper_cut(end_sign)
                bumper_name=("Front" if i==0 else "Rear")+" bumper side "+suffix
                def side_part(x,z):
                    within_end=(end_sign*(x-cut[-1][0])>=0)
                    seam=smooth(sorted(cut),x)
                    return bumper_name if within_end and z<seam else name+" "+suffix
                perforated(name+" "+suffix,physical,[],wing,paint,body,(0,s,0),
                           .021,grid_quads=True,design_normal=wing_normal,
                           partitions=[open_contour(cut,.014)],part_at=side_part)
            else:
                obj=patch(name+" "+suffix,panel,rows,cols,paint,body,(0,s,0))
                surface_normals(obj,grid,lambda q,z,s=s:skin_side(q,z,s),(0,s,0))
            # A separate return keeps the vertical lip normal out of the painted sheet.
            for end in (0,1):
                if (i==0 and end==0) or (i==3 and end==1):
                    continue
                def hem(u,v):
                    x,y,z=panel(end,u)
                    return x,y-s*.010*v,z
                patch(name+" folded edge "+suffix,hem,cols,1,body_dark,body)
        for which,arc in enumerate(arches):
            def return_surface(u,v):
                q,z=arc[round(u*(len(arc)-1))]
                x,y,z=skin_side(q,z,s)
                return x,y-s*.245*v,z
            patch(("Front" if which==0 else "Rear")+" arch return "+suffix,
                  return_surface,len(arc)-1,3,black,body)
            outline=[return_surface(k/(len(arc)-1),1) for k in range(len(arc))]
            mesh("Wheel well liner "+suffix,outline,[tuple(range(len(outline)))],
                 wheelwell,body,False,(0,s,0))
        for j in (1,2,3):
            # The weatherstrip is recessed behind both folded panel edges.
            coords=[]
            for k in range(61):
                t=k/60
                q=cuts[j](t)
                z=lerp(smooth(SILL,cuts[j](0)),smooth(BELT,q),t)
                coords.append(skin_side(q,z,s,-.008))
            path("Door weatherstrip "+suffix+str(j),coords,.006,black,trim)
        for x,z in ((.045,.859),(1.130,.880)):
            cup=[(-.102,-.007),(-.092,.017),(-.069,.035),(.052,.039),(.088,.032),
                 (.098,.015),(.094,-.012),(.078,-.031),(.047,-.035),(-.076,-.032)]
            arm=[(-.094,-.008),(-.080,.011),(-.058,.020),(.055,.024),(.083,.019),
                 (.084,.006),(.071,-.002),(-.056,-.003),(-.080,-.009)]
            def on_door(dx,dz,h,offset=0):
                return skin_side(x+dx,z+dz,s,offset+h)
            island("Door handle cup "+suffix,cup,on_door,paint,trim,(0,s,0),.003)
            island("Handle finger recess "+suffix,inset(cup,.91),lambda dx,dz,h:on_door(dx,dz,h,.004),
                   black,trim,(0,s,0),.001)
            island("Painted pull handle "+suffix,arm,lambda dx,dz,h:on_door(dx,dz*1.86-.006,h,.010),
                   paint,trim,(0,s,0),.016)
            if x<.1:
                ellipsoid("Handle lock",on_door(.060,.012,.026),(.007,.002,.007),chrome,trim)
        ellipsoid("Mirror mount "+suffix,(-.720,s*.699,1.066),(.070,.044,.041),black,trim)
        path("Mirror stalk "+suffix,[(-.750,s*.696,1.064),(-.748,s*.785,1.079),
                                    (-.739,s*.835,1.094)],.021,black,trim)
        def mirror_surface(u,v):
            # Rounded rectangular casting with a nearly flat lower shell.
            a=u*math.tau
            w=smooth([(0,.43),(.10,.80),(.27,.98),(.53,1),(.73,.94),(.91,.74),(1,.30)],v)
            xx=math.copysign(abs(math.cos(a))**.72,math.cos(a))
            yy=math.copysign(abs(math.sin(a))**.60,math.sin(a))
            return (-.736+.018*(v-.5)+.053*w*xx+.047*w*yy,
                    s*(.858+.009*xx+.097*w*yy),1.061+.122*v-.024*w*yy)
        for label,lo,high,finish in [("Lower mirror shell",0,.40,black),("Mirror painted cap",.40,1,paint)]:
            patch(label+" "+suffix,lambda u,v:mirror_surface(u,lerp(lo,high,v)),48,18,finish,trim)
        for v,finish in [(0,black),(1,paint)]:
            loop=[mirror_surface(k/48,v) for k in range(48)]
            mesh("Mirror end cap "+suffix,loop,[tuple(range(48))],finish,trim,normal=(0,0,1 if v else -1))
        path("Mirror waist seal "+suffix,[mirror_surface(k/64,.40) for k in range(64)],.0025,black,trim,True)
        mirror_shape=[(-.089,-.026),(-.087,.023),(-.071,.041),(.055,.045),
                      (.081,.034),(.089,-.009),(.079,-.034),(-.055,-.041)]
        def mirror_face(dy,dz,h):
            return (-.681+.047*(dy/.09)-.014*(dy/.09)**2-.007*(dz/.05)**2+h,
                    s*(.863+dy),1.123+dz-.024*dy/.097)
        island("Door mirror seal "+suffix,mirror_shape,mirror_face,black,trim,(1,0,0),.004)
        island("Door mirror glass "+suffix,inset(mirror_shape,.916),
               lambda y,z,h:mirror_face(y,z,h+.006),mirror_finish,trim,(1,0,0),.001)
    for a,b,name in [(NOSE,-1.015,"Hood outer skin"),(-1.009,TAIL,"Belt deck"),(1.716,TAIL,"Trunk outer skin")]:
        def cap(u,v):
            x=lerp(a,b,u)
            f=2*v-1
            y=flank_y(x,smooth(BELT,x))*f
            belt=smooth(BELT,x)
            crown=smooth(CROWN,x)*(1-f*f)
            if name=="Hood outer skin":
                # Loft a broad bonnet between the measured side silhouette and the
                # fender edge. Decouple height from plan so raising it doesn't move its nose.
                centre_start=shell_x(NOSE,0,nose_height(0))
                centre_end=shell_x(b,0,smooth(BELT,b)+smooth(CROWN,b))
                centre_x=lerp(centre_start,centre_end,u)
                ordinate=(1-abs(f)**3)
                front_section_z=lerp(smooth(BELT,NOSE),bonnet_profile(centre_start),ordinate)
                crown=(bonnet_profile(centre_x)-belt)*ordinate
                crown+=(nose_height(f)-front_section_z)*(1-u)**4
                # Align the bonnet's centre crown to the side tracing without
                # moving the matching bumper, wing or cowl borders.
                rise=smooth([(0,0),(.10,.006),(.29,.009),(.59,.010),(.75,.007),(1,0)],u)
                crown+=rise*ordinate
                # The pressed diagonal runs from the lamp's inside corner toward
                # the windscreen. Its highlight broadens at the cowl.
                ridge=lerp(.61,.86,u)
                crown+=.019*math.sin(math.pi*u)**.72*math.exp(-((abs(f)-ridge)/.085)**2)
            # Roll the shoulder into the upright wing. Keep the common outer edge
            # and the measured leading and trailing silhouettes fixed.
            edge_distance=(1-abs(f))*flank_y(x,belt)
            blend=min(1,max(0,(x-NOSE)/.30),max(0,(TAIL-x)/.24))
            roll=.10 if name=="Hood outer skin" else lerp(.22,.10,max(0,min(1,(x-.6)/.65)))
            crown+=roll*edge_distance*math.exp(-(edge_distance/.105)**2)*blend*blend*(3-2*blend)
            z=belt+crown
            if name=="Hood outer skin":
                root_x=lerp(fascia(-1,flank_y(NOSE,smooth(BELT,NOSE))*f,nose_height(f))[0],
                            shell_x(b,flank_y(b,smooth(BELT,b))*f,smooth(BELT,b)+smooth(CROWN,b)*(1-f*f)),u)
                # Exact shared outside edge with the upright fender.
                return lerp(root_x,shell_x(x,y,z),abs(f)**6),y,z
            return shell_x(x,y,z),y,z
        if name == "Belt deck":
            sections=[(a,-.861,"Front fender shoulder"),(-.861,.252,"Front door shoulder"),
                      (.252,1.276,"Rear door shoulder"),(1.276,b,"Rear quarter shoulder")]
            for suffix,va,vb in [("L",0,.065),("R",.935,1)]:
                for start,end,label in sections:
                    lo,hi=(start-a)/(b-a),(end-a)/(b-a)
                    count=max(3,math.ceil((end-start)/.043))
                    patch(label+" "+suffix,lambda u,v:cap(lerp(lo,hi,u),lerp(va,vb,v)),
                          count,7,paint,body,(0,0,1),.004)
        elif name=="Hood outer skin":
            def cut(u):
                return .914-.125*u+.012*math.sin(math.pi*u)
            patch(name,lambda u,v:cap(u,.5+.5*(2*v-1)*(cut(u)-.0018)),
                  60,64,paint,body,(0,0,1),.003)
            for sign in (-1,1):
                patch("Fender upper return",lambda u,v:cap(u,.5+.5*sign*lerp(cut(u)+.0018,1,v)),
                      60,16,paint,body,(0,0,1))
                def channel(u,v):
                    x,y,z=cap(u,.5+.5*sign*(cut(u)+.0035*(2*v-1)))
                    return x,y,z-.0025
                patch("Bonnet gap shadow",channel,60,2,black,trim,(0,0,1))
            for sign in (-1,1):
                x,y,z=cap(.727,.5+sign*.199)
                ellipsoid("Washer jet "+("L" if sign<0 else "R"),(x,y,z+.0015),
                          (.011,.013,.005),black,trim)
        else:
            patch(name,lambda u,v:cap(u,.5+.5*(2*v-1)*.8675),54,56,paint,body,(0,0,1),.004)
    block("Satin undertray",(-.042,0,.230),(3.82,1.28,.09),.035,black,body)


def lateral_surface(x,z,s):
    """Locate a side landmark after the plan-view bumper wrap."""
    for sign,end in [(-1,NOSE),(1,TAIL)]:
        if (x-skin_side(end,z,s)[0])*sign>0:
            lo,hi=0,flank_y(end,z)
            for _ in range(24):
                mid=(lo+hi)*.5
                if (fascia(sign,mid,z)[0]-x)*sign>0:
                    lo=mid
                else:
                    hi=mid
            return fascia(sign,s*(lo+hi)*.5,z,.0009)
    lo,hi=NOSE,TAIL
    for _ in range(28):
        mid=(lo+hi)*.5
        if skin_side(mid,z,s)[0] < x:
            lo=mid
        else:
            hi=mid
    return skin_side((lo+hi)*.5,z,s,.001)


def side_bumper_cut(sign):
    """Side-view shut line ending on a sampled point of the actual wheel arch."""
    if sign<0:
        angle=math.pi*20/80
        arch=(AXLES[0]-.387*math.cos(angle),.220+.465*math.sin(angle))
        return [(-2.35,.682),(-2.02,.681),(-1.87,.623),(-1.79,.603),arch]
    angle=math.pi*56/80
    arch=(AXLES[1]-.387*math.cos(angle),.220+.465*math.sin(angle))
    return [(2.39,.729),(2.15,.721),(2.04,.702),(1.83,.679),(1.74,.616),arch]


def quarter_lines():
    # The downloaded drawing's side crop starts at (20,77).
    def side_guides(px,s):
        pts=[(NOSE+(x-7)*4.479/965,(323-z)*1.505/315) for x,z in px]
        return [lateral_surface(x,z,s) for x,z in open_contour(pts)]
    for s in (-1,1):
        front_cut=side_bumper_cut(-1)
        visible=[(x,z) for x,z in open_contour(front_cut,.010) if x>=-1.96]
        path("Front bumper to fender",[lateral_surface(x,z,s) for x,z in visible],.0017,body_dark,trim)
        for label,guide in [("Lower side bumper crease",[(878,240),(918,236),(944,235),(963,232)])]:
            path(label,side_guides(guide,s),.0023,body_dark,trim)
        x,z=NOSE+(276-7)*4.479/965,(323-143)*1.505/315
        xx,y,zz=lateral_surface(x,z,s)
        ellipsoid("Front fender repeater bezel",(xx,y,zz),(.023,.003,.012),chrome,trim)
        ellipsoid("Clear fender repeater",(xx,y+s*.003,zz),(.017,.003,.008),lens_light,trim)
        outline=[(-.90,.336),(-.56,.342),(.45,.358),(.77,.364),(.86,.371),
                 (.79,.352),(.65,.346),(-.895,.321)]
        island("Pressed rocker molding",outline,lambda x,z,h:skin_side(x,z,s,h+.0015),paint,trim,(0,s,0),.006)


def greenhouse():
    # (x, elevation) guides digitized from the attached side and top views.
    # Crown line measured on the side photograph, with roof-edge height below the centre crown.
    side_roof=[(265,178),(342,146),(414,112),(463,95),(524,87),(609,85),
               (691,87),(747,93),(810,117),(856,140),(890,159)]
    roof_edge=fair_curve([(NOSE+(x-26)*4.479/965,(400-p)*1.505/315-.027) for x,p in side_roof])
    base_width=fair_curve([(-1.13,.704),(-.77,.744),(-.15,.751),(.79,.749),(1.39,.735),(1.78,.686)])
    roofwidth=[(-1.13,.697),(-.65,.613),(-.35,.543),(.29,.556),(.87,.549),(1.11,.548),
               (1.42,.602),(1.78,.679)]
    low=lambda x: smooth(BELT,x)+.003
    def land(x):
        """0 → 1 over the last stretch where the backlight settles onto the deck."""
        t=max(0,min(1,(x-1.50)/.28))
        return t*t*(3-2*t)
    def deck_edge(x):
        w=smooth(roofwidth,x)/flank_y(x,smooth(BELT,x))
        return smooth(BELT,x)+smooth(CROWN,x)*(1-w*w)+.002
    hi=lambda x: max(low(x)+.002,lerp(roof_edge(x),deck_edge(x),land(x)))
    def cabin_y(x,z):
        # Preserve the same rake through the pillar and the glazed opening.
        height=z-low(x)
        return base_width(x)-.457*height-.121*height*height
    def glazing(u,v):
        f=2*v-1
        # The source roof starts farther forward on the centreline than beside
        # the gutters. Trace that bow on both the glass and the painted panel.
        roof_leading_sweep=(.080+.011*abs(f)**2)*math.exp(-((u-.282)/.12)**2)
        sweep_weight=smooth([(0,0),(.10,1),(.66,1),(1,0)],u)
        x=lerp(-1.13,1.78,u)-roof_leading_sweep*sweep_weight
        bow=smooth([(0,-.078),(.28,-.055),(.77,.032),(1,.046)],u)
        # At the trunk the glass takes the deck's own crown so the two meet flush.
        w=smooth(roofwidth,x)/flank_y(x,smooth(BELT,x))
        crown=lerp(.013+.018*math.sin(math.pi*u),smooth(CROWN,x)*w*w,land(x))
        # Restore the frontal glass height over the advanced roof edge. The
        # correction decays around the crown and is zero at both side pillars.
        crown+=.024*math.exp(-((u-.272)/.060)**2)
        return (x+bow*(1-f*f), f*cabin_y(x,hi(x)), hi(x)+crown*(1-f*f))
    for name,a,b,mat in [("Windshield seal",0,.282,black),("Roof panel",.282,.774,paint),
                         ("Backlight seal",.774,1,black)]:
        if mat==paint:
            patch(name,lambda u,v:glazing(lerp(a,b,u),v),32,34,mat,cabin,(0,0,1),.005)
            def lining(u,v):
                x,y,z=glazing(lerp(a+.005,b-.005,u),lerp(.02,.98,v))
                return x,y,z-.018
            patch("Roof headliner",lining,30,32,lining_mat,cabin,(0,0,-1))
        if mat==black:
            # The frit turns through radiused corners in glass coordinates.
            pad=.033 if a==0 else .030
            def borders(v):
                bend=abs(2*v-1)**10
                start,end=(.055,.930) if a==0 else (.116,.820)
                return (start+.115*bend,end-.128*bend)
            def pane(u,v):
                lo,hi=borders(v)
                return glazing(lerp(a,b,lerp(lo,hi,u)),lerp(pad,1-pad,v))
            patch("Windshield" if a==0 else "Rear window",pane,38,64,glass_mat,cabin,(0,0,1))
            for left,right in [(0,pad),(1-pad,1)]:
                patch(name,lambda u,v:glazing(lerp(a,b,u),lerp(left,right,v)),32,9,black,cabin,(0,0,1))
            for end in (0,1):
                def band(u,v):
                    limit=borders(v)[end]
                    t=lerp(0,limit,u) if end==0 else lerp(limit,1,u)
                    return glazing(lerp(a,b,t),lerp(pad,1-pad,v))
                patch("Rounded ceramic frit",band,10,74,black,cabin,(0,0,1))
                path("Window perimeter lip",[band(0 if end==0 else 1,k/80) for k in range(81)],
                     .0033,black,trim)
    for u,name in [(0,"Windshield cowl return"),(1,"Backlight to trunk return")]:
        def closing(t,v):
            x,y,z=glazing(u,v)
            f=min(1,abs(y)/flank_y(x,smooth(BELT,x)))
            deck=smooth(BELT,x)+smooth(CROWN,x)*(1-f*f)
            return (x,y,lerp(deck-.003,z,t))
        patch(name,closing,3,74,black if u==0 else paint,cabin,(-1 if u==0 else 1,0,0))
    def wiper_point(u,v, lift):
        x,y,z=glazing(u,v)
        return x-.003,y,z+lift
    for i,(pivot,tip) in enumerate(((.13,.51),(.55,.90))):
        arm=[wiper_point(.007,pivot,.007),wiper_point(.010,pivot+.061,.010),
             wiper_point(.013,pivot+.154,.010)]
        blade=[wiper_point(.013+.009*k/24,lerp(pivot+.09,tip,k/24),.010)
               for k in range(25)]
        path("Windshield wiper arm "+str(i+1),arm,.0043,black,trim)
        path("Windshield wiper blade "+str(i+1),blade,.0060,black,trim)
    outlines=[
        [(-.827,low(-.827)+.006),(-.720,hi(-.720)-.050),(-.575,hi(-.575)-.045),
         (-.381,hi(-.381)-.054),(-.164,hi(-.164)-.060),(.150,hi(.150)-.062),
         (.181,hi(.181)-.082),(.146,low(.146)+.010),(.116,low(.116)+.005),(-.69,low(-.69)+.004)],
        [(.257,low(.257)+.010),(.272,hi(.272)-.085),(.307,hi(.307)-.069),
         (.625,hi(.625)-.076),(.818,1.375),(1.003,1.278),(1.273,1.117),
         (1.250,low(1.250)+.008),(.67,low(.67)+.006)],
    ]
    upper_shut=[(1.276,1.069),(1.355,1.154),(1.366,1.204),(1.344,1.252),
                (1.268,1.314),(1.141,1.378),(.990,1.430),(.851,1.459),(.664,1.470),
                (.456,1.478),(.284,1.477)]
    front_frame=[(-.866,low(-.866)-.015)]
    front_frame += [(x,hi(x)-.012) for x in (-.858,-.74,-.62,-.46,-.30,-.12,.08,.217)]
    front_frame += [(.221,1.266),(.207,low(.207)-.015)]
    rear_frame=[(.218,low(.218)-.015),(.249,1.278),(.273,1.445)]
    rear_frame += [(x,min(z,hi(x)-.007)) for x,z in reversed(upper_shut)]
    rear_frame += [(1.276,low(1.276)-.015)]
    frame_cuts=[outline+[outline[0]] for outline in (front_frame,rear_frame)]
    for s,suffix in [(-1,"L"),(1,"R")]:
        xx=[lerp(-1.13,1.78,k/160) for k in range(161)]
        outer=[(x,low(x)) for x in xx]+[(x,hi(x)) for x in reversed(xx)]
        holes=[perimeter(points,9) for points in outlines]
        def frame_part(x,z):
            if inside((x,z),front_frame):
                return "Front door window frame "+suffix
            if inside((x,z),rear_frame):
                return "Rear door window frame "+suffix
            return "Open pillar frame "+suffix
        perforated("Open pillar frame "+suffix,outer,holes,lambda x,z:(x,s*cabin_y(x,z),z),
                   paint,cabin,(0,s,0),.026,partitions=frame_cuts,part_at=frame_part)
        for i,points in enumerate(outlines):
            offset=lambda d:lambda x,z,h:(x,s*(cabin_y(x,z)+d+h),z)
            outline=perimeter(points,9)
            inner=perimeter(inset(points,.900),9)
            annulus("Window reveal "+suffix+str(i),[offset(.001)(x,z,0) for x,z in outline],
                    [offset(-.004)(x,z,0) for x,z in inner],black,cabin,(0,s,0))
            island("Door glass "+suffix+str(i),inset(points,.902),offset(-.004),glass_mat,cabin,(0,s,0),.002)
        fixed=[(.961,1.306),(.980,1.295),(.959,1.084),(.939,1.084)]
        island("Rear fixed glass divider "+suffix,fixed,lambda x,z,h:(x,s*(cabin_y(x,z)+.001),z),
               black,cabin,(0,s,0))
        seam=[(.24,1.071),(.257,1.278),(.283,1.448)]
        path("Door frame split "+suffix,[(x,s*(cabin_y(x,z)+.004),z) for x,z in seam],
             .0025,black,trim)
        path("Rear door upper shut line "+suffix,[(x,s*(cabin_y(x,min(z,hi(x)-.004))+.0014),min(z,hi(x)-.004))
              for x,z in open_contour(upper_shut)],
             .0021,body_dark,trim)
        pillar=[(.117,1.069),(.159,1.435),(.264,1.443),(.215,1.075)]
        island("Black B pillar applique "+suffix,pillar,lambda x,z,h:(x,s*(cabin_y(x,z)+.012),z),
               pillar_black,cabin,(0,s,0))
        path("Window belt molding "+suffix,[(x,s*(cabin_y(x,low(x))+.003),low(x)+.006)
             for x in [lerp(-.998,1.220,k/65) for k in range(66)]],.0045,black,trim)
    ellipsoid("Roof aerial foot",(.831,0,1.447),(.046,.028,.015),black,trim)
    path("Roof aerial",[(.844,0,1.455),(.863,0,1.509),(.923,0,1.581),
         (1.016,0,1.674)],.0055,black,trim)


def passenger_compartment():
    interior=collection("06 • Cabin visible through glazing",car)
    upholstery=material("Charcoal woven fabric",(.037,.041,.040),.84)
    dash=material("Grained dashboard",(.030,.034,.034),.61)
    micro_bump(upholstery,310,.095,.002)
    def seat_pad(name,cx,cy,cz,w,d,h,lean=0):
        profiles=[(-1,.69),(-.91,.88),(-.70,.98),(.12,1),(.69,.96),(.93,.82),(1,.60)]
        verts=[]
        for zz,sc in profiles:
            for j in range(48):
                a=j*math.tau/48
                def rounded(t):
                    return math.copysign(abs(t)**.53,t)
                verts.append((cx+lean*zz+d*sc*rounded(math.cos(a)),
                              cy+w*sc*rounded(math.sin(a)),cz+h*zz))
        faces=[(i*48+j,i*48+(j+1)%48,(i+1)*48+(j+1)%48,(i+1)*48+j)
               for i in range(len(profiles)-1) for j in range(48)]
        faces.extend([tuple(reversed(range(48))),tuple(range(len(verts)-48,len(verts)))])
        return mesh(name,verts,faces,upholstery,interior)
    block("Centre high-mounted brake light",(1.58,0,1.102),(.075,.246,.037),.015,dash,interior)
    block("Rear brake strip",(1.617,0,1.109),(.008,.218,.013),.004,red,interior)
    block("Cabin floor",(.24,0,.39),(2.50,1.48,.12),.04,dash,interior)
    for s in (-1,1):
        for x,z in [(-.08,1.183),(.96,1.133)]:
            y=s*(.299 if x<0 else .310)
            seat_pad("Seat head restraint",x+.030,y,z,.103,.061,.076,.011)
            for offset in (-.056,.056):
                path("Headrest stem",[(x+.025,y+offset,z-.075),(x+.021,y+offset,z-.159)],.006,chrome,interior)
            seat_pad("Seat back",x-.021,y,.871,.214,.081,.236,.052)
            seat_pad("Seat cushion",x-.153,y,.608,.218,.217,.061)
        block("Cabin inner door panel",(.15,s*.735,.767),(1.99,.07,.40),.045,dash,interior)
        visor=block("Folded sun visor",(-.111,s*.299,1.421),(.13,.30,.017),.009,lining_mat,interior)
        visor.rotation_euler.y=-.12
    patch("Dashboard upper pad",lambda u,v:(lerp(-1.24,-.71,u)+.095*(2*v-1)**2,
          lerp(-.735,.735,v),.927+.091*(1-u)+.019*(1-(2*v-1)**2)),20,32,dash,interior,(0,0,1),.063)
    patch("Rear parcel shelf",lambda u,v:(lerp(1.17,1.77,u)-.064*(2*v-1)**2,
          lerp(-.719,.719,v),1.036+.025*u),16,22,upholstery,interior,(0,0,1),.018)
    center=(-.609,-.325,.945)
    def steering(r,a):
        return (center[0]-.30*r*math.sin(a),center[1]+r*math.cos(a),center[2]+r*math.sin(a))
    path("Steering wheel", [steering(.166,i*math.tau/96) for i in range(96)],.015,dash,interior,True)
    for a in (.06,math.pi-.06,math.pi*1.50):
        path("Steering wheel spoke",[steering(.020,a),steering(.160,a)],.018,dash,interior)
    ellipsoid("Wheel airbag",center,(.042,.074,.047),dash,interior)
    ellipsoid("Instrument binnacle",(-.802,-.324,.977),(.122,.174,.069),dash,interior)
    path("Interior mirror stalk",[(-.334,0,1.413),(-.411,0,1.340)],.008,dash,interior)
    block("Interior rear-view mirror",(-.421,0,1.329),(.029,.183,.054),.013,dash,interior)


def face_mapping(sign, step):
    return lambda y,z,h:fascia(sign,y,z,step+h)


def inside(point, polygon):
    x,y = point
    odd = False
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        if (a[1]>y) != (b[1]>y) and x < a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1]):
            odd = not odd
    return odd


def perforated(name, outer, holes, mapping, mat, coll, direction, spacing=.035,
               grid_quads=False,design_normal=None,partitions=(),part_at=None):
    """Constrain every opening edge in the triangulation before fitting it in 3D."""
    coords, edges = [], []
    for loop in [outer,*holes]:
        start=len(coords)
        for a,b in zip(loop,loop[1:]+loop[:1]):
            count=max(1,math.ceil(math.dist(a,b)/spacing))
            coords.extend((lerp(a[0],b[0],j/count),lerp(a[1],b[1],j/count)) for j in range(count))
        n=len(coords)-start
        edges.extend((start+i,start+(i+1)%n) for i in range(n))
    for cut in partitions:
        start=len(coords)
        coords.extend(cut)
        edges.extend((i,i+1) for i in range(start,len(coords)-1))
    lo=[min(p[i] for p in outer) for i in range(2)]
    hi=[max(p[i] for p in outer) for i in range(2)]
    def solid(p):
        return inside(p,outer) and not any(inside(p,h) for h in holes)
    for j in range(1,math.ceil((hi[1]-lo[1])/spacing)):
        for i in range(1,math.ceil((hi[0]-lo[0])/spacing)):
            stagger=0 if grid_quads else .27*(j%2)
            p=(lo[0]+(i+stagger)*spacing,lo[1]+j*spacing)
            if solid(p):
                coords.append(p)
    vs,_,triangles,*_ = delaunay_2d_cdt([Vector(p) for p in coords],edges,[],0,.000005)
    faces=[tri for tri in triangles if solid(tuple(sum(vs[i][k] for i in tri)/3 for k in range(2)))]
    if grid_quads:
        # Keep full lattice cells editable as quads. Only the trimmed strip needs triangles.
        cells={}
        boundary=[]
        for tri in faces:
            ij=[((vs[p].x-lo[0])/spacing,(vs[p].y-lo[1])/spacing) for p in tri]
            rounded=[(round(x),round(y)) for x,y in ij]
            if (max(math.dist(a,b) for a,b in zip(ij,rounded))<.0005 and
                    all(max(v[k] for v in rounded)-min(v[k] for v in rounded)==1 for k in (0,1))):
                key=(min(x for x,y in rounded),min(y for x,y in rounded))
                cells.setdefault(key,[]).append((tri,rounded))
            else:
                boundary.append(tri)
        faces=boundary
        for (x,y),group in cells.items():
            if len(group)==2:
                corners={p:index for tri,ij in group for index,p in zip(tri,ij)}
                faces.append([corners[p] for p in [(x,y),(x+1,y),(x+1,y+1),(x,y+1)]])
            else:
                faces.extend(tri for tri,_ in group)
    batches={}
    for face in faces:
        label=name
        if part_at is not None:
            label=part_at(*(sum(vs[i][axis] for i in face)/len(face) for axis in range(2)))
        batches.setdefault(label,[]).append(face)
    # Shade from the shared design surface even beside small constrained triangles.
    step=.0003
    axis=Vector(direction)
    def surface_normal(a,b):
        if design_normal:
            n=design_normal(a,b)
        else:
            da=Vector(mapping(a+step,b))-Vector(mapping(a-step,b))
            db=Vector(mapping(a,b+step))-Vector(mapping(a,b-step))
            n=da.cross(db).normalized()
        if n.dot(axis)<0:
            n.negate()
        return n
    result={}
    for label,batch in batches.items():
        used=sorted({i for face in batch for i in face})
        indices={old:new for new,old in enumerate(used)}
        points=[vs[i] for i in used]
        polygons=[[indices[i] for i in face] for face in batch]
        obj=mesh(label,[mapping(*p) for p in points],polygons,mat,coll,normal=direction)
        obj.data.normals_split_custom_set_from_vertices([surface_normal(*p) for p in points])
        result[label]=obj
    return result.get(name)


def bead(name, loop, mapping, radius, mat, coll):
    return path(name,[mapping(*p) for p in loop],radius,mat,coll,True)


def annulus(name, a, b, mat, coll, direction):
    n=len(a)
    return mesh(name,[*a,*b],[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)],
                mat,coll,normal=direction)


# Pixel guides use the doubled front crop with origin at (1080, 72) on the supplied sheet.
FRONT_CX, FRONT_FLOOR = 423.0, 652.0
PIX_Y, PIX_Z = .002421, .00242


def trace(points, sign=1):
    return [(sign*(FRONT_CX-x)*PIX_Y,(FRONT_FLOOR-y)*PIX_Z) for x,y in points]


def profiled_well(name, points, depth, lip, finish, scale=.965, backing=None):
    outer=perimeter(points,6)
    inner=perimeter(inset(points,scale),6)
    rim=[fascia(-1,y,z,.002) for y,z in outer]
    back=[fascia(-1,y,z,-depth) for y,z in inner]
    annulus(name+" • reveal",rim,back,finish,details,(-1,0,0))
    if lip:
        bead(name+" • rolled edge",outer,lambda y,z:fascia(-1,y,z,.0025),lip,finish,details)
    island(name+" • backing",inset(points,scale),face_mapping(-1,-depth-.002),
           backing or wheelwell,details,(-1,0,0))
    return outer


def honeycomb(points):
    loop=perimeter(inset(points,.958),6)
    # One continuous mesh of open extruded hexagonal cells, clipped clear of the surround.
    verts, faces=[],[]
    width,height=.041,.018
    for row in range(10):
        for col in range(-14,15):
            y=(col+.5*(row%2))*width
            z=.525+row*height*.77
            hexagon=[(y+width*a,z+height*b) for a,b in
                     [(0,-.52),(.50,-.25),(.50,.25),(0,.52),(-.50,.25),(-.50,-.25)]]
            if not all(inside(p,loop) for p in hexagon):
                continue
            inner=[(lerp(y,a,.84),lerp(z,b,.72)) for a,b in hexagon]
            start=len(verts)
            for ring,offset in [(hexagon,-.008),(inner,-.008),(inner,-.022)]:
                verts.extend(fascia(-1,a,b,offset) for a,b in ring)
            for i in range(6):
                j=(i+1)%6
                faces.extend([(start+i,start+j,start+6+j,start+6+i),
                              (start+6+i,start+6+j,start+12+j,start+12+i)])
    mesh("Main grille • open honeycomb",verts,faces,grille_mat,details,False,(-1,0,0))


LAMP_SIDE = {
    -1: [(57,271),(59,260),(65,248),(79,235),(100,227),(127,221),
         (162,213),(138,231),(117,247),(104,260),(92,269),(73,273)],
    1: [(930,199),(937,190),(952,189),(968,190),(973,201),(974,218),
        (976,240),(975,245),(966,243),(951,230),(939,212)],
}


def optical_art(name, obj, image, uv_at, side_lamp=None):
    mat=material(name,(.53,.57,.60),.43,.04,.045)
    nodes,links=mat.node_tree.nodes,mat.node_tree.links

    def project(label, mapping):
        tex=nodes.new("ShaderNodeTexImage")
        tex.image=image
        uv=obj.data.uv_layers.new(name=label)
        coordinates=nodes.new("ShaderNodeUVMap")
        coordinates.uv_map=uv.name
        links.new(coordinates.outputs["UV"],tex.inputs["Vector"])
        for loop in obj.data.loops:
            u,v=mapping(obj.data.vertices[loop.vertex_index].co)
            uv.data[loop.index].uv=(u/image.size[0],1-v/image.size[1])
        return tex.outputs["Color"]

    color=project("Optic coordinates",uv_at)
    if side_lamp is not None:
        outline=LAMP_SIDE[side_lamp]
        border=inset(outline,.92)

        def side_uv(p):
            pixel=(508.5 + p.x * 965 / 4.479, 400 - p.z * 315 / 1.505)
            if inside(pixel,border):
                return pixel
            candidates=[]
            for a,b in zip(border,border[1:]+border[:1]):
                dx,dy=b[0]-a[0],b[1]-a[1]
                weight=((pixel[0]-a[0])*dx+(pixel[1]-a[1])*dy)/(dx*dx+dy*dy)
                weight=min(1,max(0,weight))
                candidates.append((a[0]+dx*weight,a[1]+dy*weight))
            return min(candidates,key=lambda q:math.dist(pixel,q))

        side_color=project("Side lens coordinates",side_uv)
        incoming=nodes.new("ShaderNodeNewGeometry")
        local=nodes.new("ShaderNodeVectorTransform")
        local.vector_type="VECTOR"
        local.convert_from="WORLD"
        local.convert_to="OBJECT"
        links.new(incoming.outputs["Incoming"],local.inputs["Vector"])
        axes=nodes.new("ShaderNodeSeparateXYZ")
        links.new(local.outputs["Vector"],axes.inputs["Vector"])
        lateral=nodes.new("ShaderNodeMath")
        lateral.operation="ABSOLUTE"
        links.new(axes.outputs["Y"],lateral.inputs[0])
        transition=nodes.new("ShaderNodeMapRange")
        transition.interpolation_type="SMOOTHSTEP"
        transition.inputs["From Min"].default_value=.53
        transition.inputs["From Max"].default_value=.96
        links.new(lateral.outputs[0],transition.inputs["Value"])
        blend=nodes.new("ShaderNodeMixRGB")
        links.new(transition.outputs["Result"],blend.inputs[0])
        links.new(color,blend.inputs[1])
        links.new(side_color,blend.inputs[2])
        color=blend.outputs[0]

    tone=nodes.new("ShaderNodeGamma")
    tone.inputs["Gamma"].default_value=1.28
    links.new(color,tone.inputs[0])
    shader=nodes.get("Principled BSDF")
    shader.inputs["Specular IOR Level"].default_value=.15
    links.new(tone.outputs[0],shader.inputs["Base Color"])
    obj.data.materials[0]=mat


def transparent_skin(name, color, reflection):
    mat=material(name,color,.095,.18,.36)
    nodes,links=mat.node_tree.nodes,mat.node_tree.links
    clear=nodes.new("ShaderNodeBsdfTransparent")
    clear.inputs[0].default_value=(*color,1)
    mix=nodes.new("ShaderNodeMixShader")
    glossy=nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value=(.79,.85,.88,1)
    glossy.inputs["Roughness"].default_value=.145 if reflection>.01 else .073
    fresnel=nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value=1.45
    optical=nodes.new("ShaderNodeMath")
    optical.operation="MULTIPLY_ADD"
    optical.inputs[1].default_value=.31 if reflection>.01 else .14
    optical.inputs[2].default_value=reflection
    links.new(fresnel.outputs[0],optical.inputs[0])
    links.new(optical.outputs[0],mix.inputs[0])
    transmission=clear.outputs[0]
    if reflection>.01:
        tint=nodes.new("ShaderNodeBsdfDiffuse")
        tint.inputs[0].default_value=(.073,.113,.129,1)
        layer=nodes.new("ShaderNodeMixShader")
        layer.inputs[0].default_value=.15
        links.new(transmission,layer.inputs[1])
        links.new(tint.outputs[0],layer.inputs[2])
        transmission=layer.outputs[0]
    links.new(transmission,mix.inputs[1])
    links.new(glossy.outputs[0],mix.inputs[2])
    links.new(mix.outputs[0],nodes.get("Material Output").inputs["Surface"])
    return mat


def reflector_depth(y,z,side):
    depth=0
    for cx,cy,r in [(138,353,.087),(184,368,.045)]:
        py,pz=trace([(cx,cy)],side)[0]
        d=math.hypot(y-py,z-pz)/r
        if d<1:
            depth+=.032*(1-d*d)**2
    return depth


def lamp_coating(obj,side,name):
    if reference_image:
        optical_art(name,obj,reference_image,lambda p:
                    (1080+(FRONT_CX-side*p.y/PIX_Y)/2,72+(FRONT_FLOOR-p.z/PIX_Z)/2),side_lamp=-1)


def reflector_bowls(side):
    for cx,cy,r in [(138,353,.087),(184,368,.045)]:
        y0,z0=trace([(cx,cy)],side)[0]
        def dish(u,v):
            a=math.tau*u
            fraction=lerp(.085,.985,v)
            y=y0+r*fraction*math.cos(a)
            z=z0+r*fraction*math.sin(a)
            flute=.0018*(.5+.5*math.cos(22*a))*math.sin(math.pi*v)
            return fascia(-1,y,z,-.007-reflector_depth(y,z,side)+flute)
        bowl=patch("Faceted parabolic reflector",dish,112,18,chrome,details,(-1,0,0))
        lamp_coating(bowl,side,"Parabolic reflector flutes")
        band=[(y0+r*.985*math.cos(j*math.tau/96),z0+r*.985*math.sin(j*math.tau/96))
              for j in range(96)]
        bead("Reflector rim",band,lambda y,z:fascia(-1,y,z,-.0068),.0016,chrome,details)
        p=fascia(-1,y0,z0,-.026)
        ellipsoid("Reflector bulb shield",p,(.009,r*.115,r*.125),chrome,details)


def cobalt_nose():
    # Outline each opening from left to centre and back along its opposite edge.
    upper=trace([(249,345),(256,330),(266,323),(300,322),(545,322),(583,324),
                 (596,334),(601,345),(575,347),(270,347)])
    main=trace([(249,365),(300,366),(420,369),(550,366),(601,365),(596,381),
                (582,414),(572,421),(521,429),(425,435),(329,430),(281,423),
                (269,416),(257,387)])
    lower=trace([(274,488),(573,488),(568,514),(558,537),(549,544),(527,546),
                 (314,546),(296,543),(284,535)])
    lens_px=[(103,279),(119,292),(146,304),(172,314),(185,325),(195,345),
             (204,369),(215,400),(209,417),(180,414),(145,410),(119,406),
             (101,399),(88,386),(85,370),(86,331),(89,307),(95,291)]
    fog_px=[(81,463),(108,466),(150,474),(174,482),(184,495),(198,536),
            (191,540),(157,536),(105,532),(88,524),(83,503)]
    lamps=[trace(lens_px,s) for s in (-1,1)]
    fogs=[trace(fog_px,s) for s in (-1,1)]
    holes=[perimeter(p,6) for p in [upper,main,lower,*lamps,*fogs]]
    top=[(flank_y(NOSE,.922)*f,nose_height(f)) for f in [lerp(-1,1,k/100) for k in range(101)]]
    side=[(flank_y(NOSE,z),z) for z in [lerp(.922,.191,k/52) for k in range(1,53)]]
    outer=top+side+[(-side[-1][0],.191)]+[(-y,z) for y,z in reversed(side[:-1])]
    perforated("Front bumper • openings and hood landing",outer,holes,
               lambda y,z:fascia(-1,y,z),paint,body,(-1,0,0),.011)
    profiled_well("Upper grille",upper,.028,.0035,chrome,.90)
    profiled_well("Main grille",main,.043,.0038,chrome,.92)
    profiled_well("Bumper intake",lower,.048,.0032,paint,.94)
    honeycomb(main)
    for z in (.760,.774):
        path("Upper grille horizontal rib",[fascia(-1,y,z,-.011) for y in
             [lerp(-.34,.34,k/32) for k in range(33)]],.0028,grille_mat,details)
    for y in [-.184,0,.184]:
        path("Intake vertical web",[fascia(-1,y,h,-.013) for h in [.262,.386]],.0042,grille_mat,details)
    path("Intake horizontal web",[fascia(-1,y,.334,-.008) for y in
         [lerp(-.325,.325,k/56) for k in range(57)]],.0062,body_dark,details)
    for s,pts,fog in zip((-1,1),lamps,fogs):
        profiled_well("Headlamp gasket",pts,.055,.0034,black,.968)
        optic,_=island("Formed multi-reflector optical insert",inset(pts,.973),
                       lambda y,z,h:fascia(-1,y,z,-.013-reflector_depth(y,z,s)),chrome,details,(-1,0,0))
        lamp_coating(optic,s,"Headlamp • fluted reflector and indicator")
        reflector_bowls(s)
        island("Clear headlamp cover",inset(pts,.98),face_mapping(-1,-.004),
               clear_lens,details,(-1,0,0),.007)
        profiled_well("Fog recess",fog,.024,.0025,paint,.91,fog_shade)
        for pp in [[(85,489),(102,492),(113,494)],[(148,508),(160,509),(175,512)]]:
            path("Fog light shelf",[fascia(-1,y,z,-.010) for y,z in trace(pp,s)],.004,paint,details)
        q,h=trace([(130,504)],s)[0]
        ring=[(q+.062*math.cos(i*math.tau/96),h+.062*math.sin(i*math.tau/96)) for i in range(96)]
        bead("Fog lamp bezel",ring,lambda y,z:fascia(-1,y,z,-.004),.0055,chrome,details)
        small=[(q+.057*math.cos(i*math.tau/24),h+.057*math.sin(i*math.tau/24)) for i in range(24)]
        fogoptic,_=island("Fluted fog reflector",small,face_mapping(-1,-.011),chrome,details,(-1,0,0),.004)
        if reference_image:
            optical_art("Fog reflector fluting",fogoptic,reference_image,
                        lambda p:(1080+(FRONT_CX-s*p.y/PIX_Y)/2,72+(FRONT_FLOOR-p.z/PIX_Z)/2))
        island("Fog lens",small,face_mapping(-1,-.003),clear_lens,details,(-1,0,0),.006)
        # Service cap above the fog recess.
        tow=trace([(216,449),(220,440),(231,441),(238,448),(241,465),(220,466)],s)
        bead("Bumper access cap",perimeter(tow,6),lambda y,z:fascia(-1,y,z,.0006),.0011,body_dark,details)
    hood_line=[(y,z+.0003) for y,z in top[4:-4]]
    path("Hood front shut line",[fascia(-1,y,z,.001) for y,z in hood_line],.0025,body_dark,details)


def cobalt_tail():
    # Rear reference is the crop starting at (1080,511), enlarged by 1.8.
    def rear_trace(px,s=1):
        return [(s*(378-x)*.00253,(617-y)*.002515) for x,y in px]
    lamp=[(78,211),(110,211),(117,224),(129,267),(149,326),(148,332),
          (83,328),(60,325),(54,318),(53,291),(54,247),(57,232)]
    pairs=[rear_trace(lamp,s) for s in (-1,1)]
    license_recess=rear_trace([(225,395),(535,395),(540,402),(540,417),(532,461),
                               (518,480),(504,486),(456,490),(287,490),(248,486),
                               (235,477),(226,461),(223,422)])
    fs=[lerp(-1,1,k/90) for k in range(91)]
    z0,z1=smooth(SILL,TAIL),smooth(BELT,TAIL)
    edge=flank_y(TAIL,z1)
    top=[(f*edge,z1+smooth(CROWN,TAIL)*(1-f*f)) for f in fs]
    border=[(flank_y(TAIL,z),z) for z in [lerp(z1,z0,k/55) for k in range(1,56)]]
    outer=top+border+[(-border[-1][0],z0)]+[(-y,z) for y,z in reversed(border[:-1])]
    bumper_height=[(0,.596),(.30,.599),(.48,.614),(.60,.652),
                   (.72,.708),(.80,.733),(.91,.738)]
    lid_width=[(.615,.40),(.647,.509),(.725,.568),(.800,.604),
               (.917,.639),(1.005,.657),(1.11,.657)]
    rear_partitions=[
        [(y,smooth(bumper_height,abs(y))) for y in (lerp(-.92,.92,k/140) for k in range(141))]
    ]
    for sign in (-1,1):
        rear_partitions.append([(sign*smooth(lid_width,z),z)
                                for z in (lerp(.596,1.11,k/56) for k in range(57))])
    def rear_part(y,z):
        if z<smooth(bumper_height,abs(y)):
            return "Rear bumper outer skin"
        if abs(y)<smooth(lid_width,z):
            return "Trunk lid rear facing"
        return "Rear lamp surround "+("L" if y<0 else "R")
    perforated("Rear bumper outer skin",outer,[perimeter(o,6) for o in [*pairs,license_recess]],
               lambda y,z:fascia(1,y,z),paint,body,(1,0,0),.014,
               partitions=rear_partitions,part_at=rear_part)
    lp=perimeter(license_recess,6)
    lpi=perimeter(inset(license_recess,.91),6)
    annulus("Rear plate pocket walls",[fascia(1,y,z) for y,z in lp],
            [fascia(1,y,z,-.025) for y,z in lpi],paint,details,(1,0,0))
    island("Rear plate pocket back",inset(license_recess,.92),face_mapping(1,-.026),paint,details,(1,0,0))
    garnish=rear_trace([(226,395),(534,395),(534,410),(225,410)])
    island("Rear chrome license molding",garnish,face_mapping(1,.002),chrome,details,(1,0,0),.002)
    for s,pts in zip((-1,1),pairs):
        outline=perimeter(pts,6)
        inner=perimeter(inset(pts,.971),6)
        annulus("Tail lamp gasket",[fascia(1,y,z,.001) for y,z in outline],
                [fascia(1,y,z,-.006) for y,z in inner],black,details,(1,0,0))
        lens,_=island("Tail light • wrapped reflector",inset(pts,.976),face_mapping(1,-.004),
                      red,details,(1,0,0),.007)
        if reference_image:
            optical_art("Tail lamp optical chambers",lens,reference_image,
                        lambda p:(1080+(378-s*p.y/.00253)/1.8,511+(617-p.z/.002515)/1.8),
                        side_lamp=1)
        island("Rear lamp clear cover",inset(pts,.974),face_mapping(1,.002),clear_lens,details,(1,0,0),.009)
    seam=rear_trace([(117,216),(118,220),(120,230),(131,270),(150,330),
                     (160,350),(175,363),(193,367),(264,370),(378,372)])
    seam=seam+[(-y,z) for y,z in reversed(seam[:-1])]
    path("Trunk perimeter",[fascia(1,y,z,.002) for y,z in open_contour(seam)],.0028,black,details)
    # One joint runs from wheel opening to wheel opening. Use a vertex on the
    # lamp gasket as its datum, then turn underneath the optical chambers.
    lamp_y,lamp_z=rear_trace([lamp[7]])[0]
    lamp_x=fascia(1,lamp_y,lamp_z)[0]
    arch_angle=math.pi*56/80
    start_x=AXLES[1]-.387*math.cos(arch_angle)
    start_z=.220+.465*math.sin(arch_angle)
    def quarter_joint(side):
        guides=[(start_x,start_z),(lerp(start_x,lamp_x,.13),start_z+.008),
                (lerp(start_x,lamp_x,.36),start_z+.074),
                (lerp(start_x,lamp_x,.70),start_z+.080),
                (lerp(start_x,lamp_x,.86),start_z+.096),(lamp_x,lamp_z)]
        run=[lateral_surface(x,z,side) for x,z in open_contour(guides,.006)]
        run[-1]=fascia(1,side*lamp_y,lamp_z,.0011)
        return run
    edge=rear_trace([lamp[7],(79,331),(108,335),(142,340),(170,361),
                     (194,373),(260,378),(378,380)])
    edge+= [(-y,z) for y,z in reversed(edge[:-1])]
    across=[fascia(1,y,z,.0011) for y,z in open_contour(edge,.006)]
    path("Rear bumper continuous shut line",quarter_joint(1)[:-1]+across+
         list(reversed(quarter_joint(-1)[:-1])),.0019,body_dark,details)
    for text,y in [("C O B A L T",-.48),("1.5 LTZ",.48)]:
        z=.757
        text_obj(text,(fascia(1,y,z,.006)[0],y,z),1,.031,chrome)
    y,z=.009,.807
    x=fascia(1,y,z,.003)[0]
    ellipsoid("Trunk cylinder",(x,y,z),(.006,.014,.014),chrome,details)
    ellipsoid("Trunk cylinder center",(x+.004,y,z),(.004,.008,.008),black,details)


def face_details():
    cobalt_nose()
    cobalt_tail()
    for s,z in [(-1,.415),(1,.425)]:
        x=fascia(s,0,z,.005 if s<0 else -.016)[0]
        block("License plate frame",(x,0,z),(.014,.562 if s<0 else .575,.137 if s<0 else .150),.006,black,details)
        block("License plate bright edge",(x+s*.006,0,z),(.009,.549 if s<0 else .566,.130 if s<0 else .143),.004,silver,details)
        block("License plate",(x+s*.010,0,z),(.009,.538 if s<0 else .557,.124 if s<0 else .132),.003,plate,details)
        text_obj("COBALT",(x+s*.016,0,z-.002),s,.135 if s<0 else .136,black)
    emblem=[(-.092,.008),(-.037,.008),(-.037,.026),(.037,.026),(.037,.008),(.093,.008),
            (.083,-.009),(.035,-.009),(.035,-.025),(-.035,-.025),(-.035,-.009),(-.095,-.009)]
    for s,z in [(-1,.708),(1,.922)]:
        x=fascia(s,0,z,.012)[0]
        badge=[(y,z+dy) for y,dy in emblem]
        if s<0:
            badge=trace([(380,369),(387,353),(404,353),(408,346),(442,346),(445,353),
                         (467,353),(460,369),(444,369),(442,379),(404,379),(402,369)])
        border=perimeter(badge,3)
        annulus("Bowtie backing",[(x-s*.009,y,h) for y,h in border],[(x,y,h) for y,h in border],black,details,(s,0,0))
        mesh("Chevrolet bowtie bright surround",[(x,y,h) for y,h in border],[tuple(range(len(border)))],
             chrome,details,False,(s,0,0))
        bead("Bowtie rolled chrome",border,lambda y,h:(x+s*.001,y,h),.0021,chrome,details)
        gold_shape=inset(badge,.82)
        gold_rim=perimeter(gold_shape,3)
        annulus("Bowtie gold bevel",[(x+s*.001,y,h) for y,h in border],
                [(x+s*.004,y,h) for y,h in gold_rim],chrome,details,(s,0,0))
        mesh("Chevrolet bowtie gold insert",[(x+s*.0045,y,h) for y,h in gold_shape],
             [tuple(range(len(gold_shape)))],gold,details,False,(s,0,0))
    outlet=fascia(1,-.535,.225,-.029)[0]
    lathe("Exhaust outlet",(outlet,-.535,.225),
          [(-.103,.036),(.007,.036),(.011,.033),(.011,.028),(-.103,.028)],
          rotor,details,axis=0,n=56)
    ellipsoid("Dark exhaust bore",(outlet,-.535,.225),(.003,.028,.028),wheelwell,details)


def text_obj(label, position, sign, size, mat):
    data=bpy.data.curves.new(label,"FONT")
    data.body=label
    if bold_font:
        data.font=bold_font
    data.align_x=data.align_y="CENTER"
    data.size=size
    data.extrude=.0004
    obj=bpy.data.objects.new(label,data)
    details.objects.link(obj)
    obj.location=position
    normal=Vector((sign,0,0))
    obj.rotation_euler=normal.to_track_quat("Z","Y").to_euler()
    # Local X follows the image horizontal and local Y points up.
    obj.rotation_euler=(math.pi/2,0,sign*math.pi/2)
    data.materials.append(mat)


def wheels():
    for index,x in enumerate(AXLES):
        for s,label in [(-1,"L"),(1,"R")]:
            coll=collection(("Front" if index==0 else "Rear")+" wheel "+label,wheel_group)
            track=.768
            hub=(x,s*track,HUB_Z)
            def turned(profile,mat,name):
                return lathe(name,hub,[(s*p,r) for p,r in profile],mat,coll)
            turned([(-.102,.222),(-.109,.258),(-.104,.291),(-.085,.312),(-.058,.324),
                    (-.045,.326),(-.040,.318),(-.035,.326),(-.005,.326),
                    (0,.317),(.005,.326),(.035,.326),(.040,.318),(.045,.326),
                    (.058,.325),(.078,.319),(.097,.305),(.109,.274),(.106,.224),
                    (.092,.203),(-.099,.203)],rubber,"Tire carcass")
            verts,faces=[],[]
            for j in range(72):
                angle=math.tau*j/72
                for direction in (-1,1):
                    start=len(verts)
                    for dy,shift,r in [(.051,.00,.326),(.075,.018,.321),(.089,.030,.311)]:
                        for offset in (-.006,.006):
                            a=angle+direction*shift+offset
                            verts.append((x+r*math.cos(a),hub[1]+dy*direction,HUB_Z+r*math.sin(a)))
                    faces.extend([(start,start+1,start+3,start+2),(start+2,start+3,start+5,start+4)])
            mesh("Tread shoulder sipes",verts,faces,tread_mat,coll,False)
            for r in (.233,.270):
                turned([(.106,r-.0013),(.110,r-.0013),(.110,r+.0013),(.106,r+.0013)],rubber,"Raised sidewall rib")
            for groove in [-.040,0,.040]:
                turned([(groove-.003,.319),(groove+.003,.319),
                        (groove+.003,.3205),(groove-.003,.3205)],tread_mat,"Tread circumferential channel")
            rim_scale=.954
            rim=[(-.070,.176),(-.072,.218),(.085,.218),(.102,.222),(.112,.218),
                 (.113,.208),(.109,.203),(.104,.196),(.086,.191),(-.065,.191)]
            turned([(axial,radius*rim_scale) for axial,radius in rim],silver,"Alloy barrel and turned lip")
            turned([(-.010,.022),(-.010,.159),(.003,.159),(.004,.022)],rotor,"Brake rotor")
            block("Brake caliper",(x+.127,s*(track+.003),HUB_Z+.069),(.085,.082,.094),.014,grille_mat,coll)
            for i in range(10):
                angle=math.pi/10+math.tau*(i//2)/5+(-.266 if i%2==0 else .266)
                outline=[(.046,-.012),(.091,-.015),(.193,-.016),(.211,-.017),(.211,.015),
                         (.154,.015),(.093,.014),(.048,.013)]
                def spoke(r,t,y):
                    r,t=r*rim_scale,t*rim_scale
                    return (x+r*math.cos(angle)-t*math.sin(angle),s*(track+y),
                            HUB_Z+r*math.sin(angle)+t*math.cos(angle))
                verts=[spoke(r,t,y+(r-.05)*.03) for y in (.072,.097) for r,t in outline]
                n=len(outline)
                faces=[tuple(reversed(range(n))),tuple(range(n,2*n))]
                faces.extend((j,(j+1)%n,(j+1)%n+n,j+n) for j in range(n))
                obj=mesh("Cast spoke "+str(i+1),verts,faces,silver,coll,False)
                b=obj.modifiers.new("Cast edge radius","BEVEL")
                b.width=.006
                b.segments=3
                obj.modifiers.new("Spoke weighted normals","WEIGHTED_NORMAL")
            turned([(.072,.007),(.072,.060),(.095,.067),(.109,.057),(.110,.007)],silver,"Hub cap")
            for i in range(5):
                angle=math.pi/2+math.tau*i/5
                lx,lz=x+.045*math.cos(angle),HUB_Z+.045*math.sin(angle)
                ellipsoid("Recessed lug seat",(lx,s*(track+.109),lz),(.015,.003,.015),wheelwell,coll)
                ellipsoid("Lug nut",(lx,s*(track+.113),lz),(.008,.004,.008),chrome,coll)
            ellipsoid("Center cap",(x,s*(track+.117),HUB_Z),(.023,.005,.023),silver,coll)
            ellipsoid("Gold wheel medallion",(x,s*(track+.123),HUB_Z),(.012,.002,.012),gold,coll)


def camera(name, position, target, ortho):
    data=bpy.data.cameras.new(name)
    data.type="ORTHO"
    data.ortho_scale=ortho
    obj=bpy.data.objects.new(name,data)
    studio.objects.link(obj)
    obj.location=position
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()
    return obj


def light(name,position,target,energy,size,size_y,color):
    data=bpy.data.lights.new(name,"AREA")
    data.energy, data.color, data.shape=energy,color,"RECTANGLE"
    data.size,data.size_y=size,size_y
    obj=bpy.data.objects.new(name,data)
    studio.objects.link(obj)
    obj.location=position
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()


def studio_sweep():
    # A circular cove gives the orthographic cameras the same floor as the hero cameras.
    v,f=[],[]
    rings=[(14.75,-3),(14.75,-.007),(11.6,-.004),(11.6,-.002)]
    rings.extend((11.6+3.2*math.sin(math.pi*j/48),3.198-3.2*math.cos(math.pi*j/48))
                 for j in range(1,25))
    rings.append((14.8,15))
    for radius,z in rings:
        for j in range(128):
            a=math.tau*j/128
            v.append((radius*math.cos(a),radius*math.sin(a),z))
    for i in range(len(rings)-1):
        for j in range(128):
            k=(j+1)%128
            f.append((i*128+j,(i+1)*128+j,(i+1)*128+k,i*128+k))
    wall=material("Cove • neutral horizon",(.36,.36,.36),.70)
    nodes,links=wall.node_tree.nodes,wall.node_tree.links
    lightpath=nodes.new("ShaderNodeLightPath")
    paper=nodes.new("ShaderNodeEmission")
    paper.inputs[0].default_value=(.69,.70,.71,1)
    mix=nodes.new("ShaderNodeMixShader")
    links.new(lightpath.outputs["Is Camera Ray"],mix.inputs[0])
    links.new(nodes.get("Principled BSDF").outputs[0],mix.inputs[1])
    links.new(paper.outputs[0],mix.inputs[2])
    links.new(mix.outputs[0],nodes.get("Material Output").inputs["Surface"])
    return mesh("Cyclorama • curved neutral wall",v,f,wall,studio)


def calibrate_vehicle(root):
    bpy.context.view_layer.update()
    graph=bpy.context.evaluated_depsgraph_get()
    mins=[float("inf")]*3
    maxs=[-float("inf")]*3
    parts=[o for o in car.all_objects if o.get("body_envelope") or o.name=="Roof panel"]
    for obj in parts:
        evaluated=obj.evaluated_get(graph)
        geometry=evaluated.to_mesh()
        for vertex in geometry.vertices:
            p=evaluated.matrix_world@vertex.co
            for k in range(3):
                mins[k]=min(mins[k],p[k])
                maxs[k]=max(maxs[k],p[k])
        evaluated.to_mesh_clear()
    root.scale=(4.479/(maxs[0]-mins[0]),1.735/(maxs[1]-mins[1]),1.505/maxs[2])
    root.location.x=-(mins[0]+maxs[0])*.5*root.scale.x

def organize_assemblies(root):
    """Expose independently editable parts with a pivot for each assembly."""
    for obj in body.objects:
        obj["body_envelope"]=True
    bpy.context.view_layer.update()
    parts=list(car.all_objects)
    assembly_root=collection("Qismlar • editable assemblies",car)
    groups={}
    fixed_rules=(
        ("01 Kapot",("Hood outer skin","Washer jet","Bonnet gap shadow","Hood front shut line")),
        ("02 Old bamper",("Front bumper","Bumper access","Bumper intake","Intake ","Fog recess",
                          "Fog light shelf","Fog lamp bezel","Fluted fog reflector","Fog lens")),
        ("03 Old panjaralar",("Upper grille","Main grille")),
        ("12 Tom va ustunlar",("Roof ","Open pillar frame","Black B pillar","Door frame split",
                              "Upper door shoulder","Rear door upper shut line")),
        ("13 Old oyna",("Windshield","Folded sun visor")),
        ("14 Orqa oyna",("Rear window","Backlight")),
        ("15 Bagaj qopqog‘i",("Trunk ","C O B A L T","1.5 LTZ")),
        ("16 Orqa bamper",("Rear bumper","Lower side bumper crease","Rear plate pocket",
                          "Rear chrome license")),
        ("23 Ostki qism va egzoz",("Satin undertray","Exhaust","Dark exhaust","Wheel well liner")),
    )

    def destination(obj,centre):
        name=obj.name
        x,y,_=centre
        side="chap" if y<0 else "o‘ng"
        if obj.users_collection[0].name.startswith(("Front wheel","Rear wheel")):
            front=x<0
            angle="old" if front else "orqa"
            axle=AXLES[0] if front else AXLES[1]
            return f"22 G‘ildirak · {angle} {side}",(axle,math.copysign(.768,y),HUB_Z)
        if name.startswith(("Headlamp","Clear headlamp","Formed multi-reflector","Faceted parabolic",
                            "Reflector rim","Reflector bulb shield")):
            return f"04 Old fara · {side}",None
        if name.startswith(("Tail lamp","Tail light","Rear lamp clear")):
            return f"17 Orqa chiroq · {side}",None
        if name.startswith(("Mirror","Lower mirror","Door mirror")):
            return f"11 Ko‘zgu · {side}",None
        if name.startswith(("Front fender","Fender upper return","Clear fender","Front arch")):
            return f"05 Old qanot · {side}",None
        if name.startswith(("Rear quarter","Rear lamp surround","Rear arch")):
            return f"18 Orqa qanot · {side}",None
        if name.startswith(("Front door","Rear door L","Rear door R","Rear door folded","Rear door window",
                            "Rear door shoulder","Door handle",
                            "Handle finger","Painted pull","Handle lock","Door glass","Window reveal",
                            "Rear fixed glass")):
            handle=name.startswith(("Door handle","Handle finger","Painted pull","Handle lock"))
            fore=x<(.2 if handle else 0)
            axis=-.89 if fore else .24
            return f"{'06 Old' if fore else '08 Orqa'} eshik · {side}",(axis,math.copysign(.72,y),.72)
        if name.startswith(("Pressed rocker","Door weatherstrip","Window belt molding")):
            return f"10 Ostonalar va rezinalar · {side}",None
        if name.startswith(("Rounded ceramic frit","Window perimeter lip")):
            return ("13 Old oyna" if x<0 else "14 Orqa oyna"),None
        if name.startswith(("License plate","COBALT")):
            return ("02 Old bamper" if x<0 else "16 Orqa bamper"),None
        if name.startswith(("Bowtie","Chevrolet bowtie")):
            return ("03 Old panjaralar" if x<0 else "15 Bagaj qopqog‘i"),None
        for label,prefixes in fixed_rules:
            if name.startswith(prefixes):
                return label,None
        return "24 Salon",None

    for obj in parts:
        if obj==root:
            continue
        centre=sum((obj.matrix_world@Vector(v) for v in obj.bound_box),Vector())/8
        label,pivot=destination(obj,centre)
        groups.setdefault(label,[]).append((obj,centre,pivot))
    pivots={"01 Kapot":(-1.04,0,1.04),"15 Bagaj qopqog‘i":(1.76,0,1.10)}
    for label,entries in sorted(groups.items()):
        group=collection(label,assembly_root)
        pivot=bpy.data.objects.new(label.split(" ",1)[1],None)
        group.objects.link(pivot)
        position=pivots.get(label)
        if position is None:
            position=next((datum for _,_,datum in entries if datum is not None),None)
        if position is None:
            position=sum((centre for _,centre,_ in entries),Vector())/len(entries)
        pivot.parent=root
        pivot.location=position
        pivot.empty_display_type="PLAIN_AXES"
        pivot.empty_display_size=.13
        pivot["assembly"]=label
        for obj,_,_ in entries:
            move(obj,group)
            obj.parent=pivot
            obj.location-=Vector(position)
    for old in (body,cabin,trim,details,wheel_group):
        for child in list(old.children):
            bpy.data.collections.remove(child)
        bpy.data.collections.remove(old)


args=options()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene
scene.unit_settings.system="METRIC"
scene["Source"] = "User-supplied Chevrolet Cobalt 2014–2020 four-view blueprint"
scene["Measurement basis"] = "4.479 m length × 1.735 m body width × 1.505 m roof target (mirrors excluded)"
car=collection("COBALT • Exterior")
body=collection("01 • Body panels and undertray",car)
cabin=collection("02 • Roof and glazing",car)
trim=collection("03 • Door trim and mirrors",car)
details=collection("04 • Bumpers, lights, and badges",car)
wheel_group=collection("05 • Wheel assemblies",car)
studio=collection("Studio • lights and cameras")
paint=material("Champagne • clearcoat",(.292,.222,.163),.35,.53,.28)
micro_bump(paint,465,.055,.001)
body_dark=material("Dark body inserts",(.116,.084,.055),.40,.39,.20)
fog_shade=material("Sculpted fog pocket inner paint",(.211,.153,.108),.40,.44,.20)
black=material("EPDM and inner plastic",(.015,.018,.020),.40)
pillar_black=material("B pillar satin",(.009,.011,.013),.43,.05,.09)
lining_mat=material("Graphite fabric headliner",(.063,.066,.065),.86)
mirror_finish=material("Tinted mirror reflector",(.092,.119,.131),.16,.91,.10)
grille_mat=material("Graphite grille polymer",(.011,.012,.013),.50,.09)
wheelwell=material("Recess and wheel tub",(.004,.005,.006),.81)
glass_mat=transparent_skin("Automotive solar glass",(.66,.75,.77),.013)
clear_lens=transparent_skin("Clear polycarbonate",(.96,.98,.99),.008)
chrome=material("Polished chrome",(.71,.75,.78),.17,.95)
silver=material("Machined alloy",(.56,.58,.60),.27,.71,.11)
rotor=material("Cast steel",(.27,.28,.29),.39,.77)
rubber=material("Tire sidewall",(.012,.014,.016),.59)
micro_bump(rubber,135,.10,.004)
tread_mat=material("Tread groove shadow",(.009,.011,.013),.68)
red=material("Ruby optical plastic",(.43,.018,.027),.225,.18,.47)
red_light=material("Tail lamp red reflector",(.70,.080,.052),.24,.28,.36)
lamp_base=material("Dark headlamp optic shell",(.096,.115,.130),.204,.33,.47)
lens_light=material("Reflective lens optic",(.46,.54,.59),.245,.57,.35)
light_silver=material("Silver reflector strip",(.70,.75,.79),.27,.42,.26)
gold=material("Gold emblem",(.79,.43,.068),.27,.65,.30)
plate=material("Off white plate enamel",(.80,.83,.79),.31,.08)
floor_mat=material("Studio floor",(.35,.355,.36),.64,.01,emission=.23)
bold_path=Path("C:/Windows/Fonts/arialbd.ttf")
bold_font=bpy.data.fonts.load(str(bold_path)) if bold_path.is_file() else None
if bold_font:
    bold_font.pack()
reference_image=None
if args.reference:
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    reference_image=bpy.data.images.load(str(args.reference.resolve()))
    reference_image.pack()
else:
    library=ROOT/"Chevrolet_Cobalt.blend"
    if library.is_file():
        with bpy.data.libraries.load(str(library),link=False) as (available,requested):
            requested.images=[name for name in available.images if name.endswith(".png")][:1]
        if requested.images:
            reference_image=requested.images[0]
    if not reference_image:
        raise ValueError("Pass --reference with the original four-view Cobalt image.")
exterior()
quarter_lines()
greenhouse()
passenger_compartment()
face_details()
wheels()

vehicle_root=bpy.data.objects.new("Cobalt • vehicle datum",None)
car.objects.link(vehicle_root)
vehicle_root.empty_display_size=.20
for obj in list(car.all_objects):
    if obj!=vehicle_root:
        obj.parent=vehicle_root
# Calibrate the modeled fender-to-fender envelope to the supplied front view.
organize_assemblies(vehicle_root)
calibrate_vehicle(vehicle_root)

if reference_image:
    ref=collection("Reference • supplied four-view drawing")
    image=reference_image
    empty=bpy.data.objects.new("Cobalt blueprint drawing",None)
    ref.objects.link(empty)
    empty.empty_display_type="IMAGE"
    empty.data=image
    empty.empty_display_size=7.0
    empty.color[3]=.45
    empty.location=(0,3,1.6)
    empty.rotation_euler.x=math.pi/2
    empty.hide_render=True
    empty.hide_set(True)

bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.002))
plane=move(bpy.context.object,studio)
plane.name="Studio seamless floor"
plane.data.materials.append(floor_mat)
cove=studio_sweep()
elevation_backdrop=cove.data.materials[0]
cove.data.materials[0]=floor_mat
world=bpy.data.worlds.new("Studio ambient")
world.use_nodes=True
world.node_tree.nodes["Background"].inputs[0].default_value=(.29,.36,.47,1)
world.node_tree.nodes["Background"].inputs[1].default_value=.42
scene.world=world
light("Key • long front softbox",(-3.4,-4.3,5.6),(-.45,0,.5),880,5.6,2.5,(1,.94,.84))
light("Fill • long side strip",(.7,4.2,4.0),(.2,0,.65),850,6.0,1.9,(.83,.91,1))
light("Rim • overhead strip",(3.4,-2.3,5.1),(.45,0,.5),510,3.7,1.0,(1,.96,.88))
light("Frontal fill",(-4.6,2.6,3.5),(-1.8,0,.59),148,1.7,2.5,(.87,.94,1))
hero=camera("01 • Front three quarter",(-6.8,-5.4,2.42),(-.05,0,.70),5.12)
rear=camera("02 • Rear three quarter",(6.7,-6.2,2.94),(0,0,.71),5.78)
side=camera("03 • Side orthographic",(0,-8,.74),(0,0,.74),5.05)
front=camera("04 • Front orthographic",(-8,0,.75),(0,0,.75),2.25)
back=camera("06 • Rear orthographic",(8,0,.75),(0,0,.75),2.25)
top=camera("05 • Top orthographic",(0,0,9),(0,0,.75),5.03)
top.rotation_euler.z=0
scene.camera=hero
scene.render.engine="CYCLES"
scene.cycles.samples=5 if args.draft else 96
scene.cycles.adaptive_threshold=.08 if args.draft else .012
scene.cycles.use_denoising=True
scene.render.resolution_x=920 if args.draft else 1560
scene.render.resolution_y=590 if args.draft else 1010
scene.render.resolution_percentage=100
scene.render.image_settings.file_format="PNG"
scene.view_settings.view_transform="AgX"
args.output=args.output.resolve()
args.output.mkdir(exist_ok=True,parents=True)

# Store the clean perspective as the first editing view.
for obj in bpy.context.view_layer.objects:
    obj.select_set(False)
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=="VIEW_3D":
            space=area.spaces.active
            space.region_3d.view_rotation=hero.rotation_euler.to_quaternion()
            space.region_3d.view_distance=7.0
            space.region_3d.view_location=(0,0,.7)
            space.overlay.show_extras=False
            space.shading.type="MATERIAL"
            space.shading.use_scene_lights=True
            space.shading.use_scene_world=True
            space.shading.color_type="MATERIAL"
            space.shading.studiolight_rotate_z=.4
if not args.draft:
    bpy.context.preferences.filepaths.save_version=0
    scene.render.filepath="//Cobalt_front.png"
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output/"Chevrolet_Cobalt.blend"))

views = [(hero,"front"), (side,"side"), (rear,"rear"), (front,"straight"), (top,"top")] if args.draft else [(hero,"front"),(rear,"rear"),(front,"straight"),(side,"side"),(back,"back"),(top,"top")]
for view,label in views:
    scene.camera=view
    cove.data.materials[0]=elevation_backdrop if view in (front,back,side) else floor_mat
    if view in (front,back):
        scene.render.resolution_x=620 if args.draft else 1010
    else:
        scene.render.resolution_x=920 if args.draft else 1560
    scene.render.filepath=str(args.output/(("draft_" if args.draft else "Cobalt_")+label+".png"))
    bpy.ops.render.render(write_still=True)
