"""Fit the Cobalt to the supplied FBX while retaining the Blender studio and part pivots."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def new_collection(name, parent):
    collection = bpy.data.collections.new(name)
    parent.children.link(collection)
    return collection


def face_islands(mesh):
    parents = list(range(len(mesh.vertices)))

    def root(index):
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for edge in mesh.edges:
        a, b = edge.vertices
        parents[root(a)] = root(b)
    islands = defaultdict(list)
    for face in mesh.polygons:
        islands[root(face.vertices[0])].append(face)
    return islands.values()


def centre_of(points):
    return Vector(tuple(
        (min(point[axis] for point in points) + max(point[axis] for point in points)) / 2
        for axis in range(3)
    ))


def finish(name, color, metallic=0, roughness=.35, coat=0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Coat Weight"].default_value = coat
    return mat


def shell_group(label, side):
    return f"{label} · {side}"


def assembly(source, point, points, source_slots):
    x, y, z = point
    side = "chap" if x < 0 else "o‘ng"
    if source == "chassis":
        if "primary" in source_slots:
            if y > 1.05 and abs(x) > .63:
                return shell_group("05 Old qanot", side), "Old qanot"
            if abs(x) > .74 and z < .22:
                return shell_group("18 Orqa qanot", side), "Yon kuzov va orqa qanot"
            return "12 Tom va ustunlar", "Tom va ustunlar"
        if z < -.24:
            return "23 Ostki qism va egzoz", "G‘ildirak ichki ravog‘i va taglik"
        if 1.29 < y < 1.58 and z < .31:
            return "13 Old oyna", "Oyna tozalagichi va tutashuv"
        return "12 Tom va ustunlar", "Ustun qoplamasi"
    if source.startswith("door_"):
        front = source.startswith(("door_lf","door_rf"))
        if front and max(abs(p.x) for p in points) > 1.06 and min(p.y for p in points) > .65:
            return shell_group("11 Ko‘zgu", side), "Ko‘zgu korpusi"
        label = "06 Old eshik" if front else "08 Orqa eshik"
        return shell_group(label, side), label[3:]
    if source.startswith("bonnet"):
        return "01 Kapot", "Kapot"
    if source.startswith("boot"):
        return "15 Bagaj qopqog‘i", "Bagaj qopqog‘i"
    if source == "bump_front_ok" and z > -.32:
        return "03 Old panjaralar", "Panjara va emblema"
    if source.startswith("bump_front") or source == "lamp_01":
        return "02 Old bamper", "Old bamper va tuman chirog‘i"
    if source.startswith("bump_rear"):
        return "16 Orqa bamper", "Orqa bamper"
    if source == "fara":
        return shell_group("04 Old fara", side), "Old fara"
    if source.startswith("Fender_") or source == "lamp_02":
        return shell_group("05 Old qanot", side), "Qanot burilish chirog‘i"
    if source in {"ascende","lanterna","lanterna1","lanternaa","lamp"}:
        return shell_group("17 Orqa chiroq", side), "Orqa chiroq"
    if source == "windscreen_ok":
        return ("13 Old oyna","Old oyna") if y > 0 else ("14 Orqa oyna","Orqa oyna")
    if source in {"freio","luzdefreio"}:
        return "14 Orqa oyna", "Markaziy stop chirog‘i"
    if source.startswith("roda"):
        axle = "old" if y > 0 else "orqa"
        return f"22 G‘ildirak · {axle} {side}", "G‘ildirak"
    if source == "escape":
        return "23 Ostki qism va egzoz", "Egzoz"
    return "24 Salon", "Salon"


def material_for(source, slot, index, point, points, label):
    _, _, z = point
    if source.startswith("roda"):
        return (dark,gold,rubber,alloy,alloy,steel)[index]
    if label.startswith("11 ") and slot != "primary":
        # The flat rear-facing component of each mirror is the reflective face.
        return chrome if max(p.y for p in points) < .855 and min(abs(p.x) for p in points) > .88 else dark
    if slot == "primary":
        return paint
    if slot.startswith("glass"):
        return window if source.startswith(("door","windscreen")) else lens
    if source == "fara":
        if slot == "fara.4":
            return lens
        if "front light" in slot or slot == "fara.3":
            return chrome
        return optic_black
    if source == "lanternaa":
        return dark
    if source == "lanterna1":
        return red if z < .11 or (max(p.z for p in points)-min(p.z for p in points)) > .23 else silver_light
    if source == "ascende":
        if z > .19:
            return silver_light
        if .11 < z < .20:
            return amber
        return red
    if source in {"lamp","freio","luzdefreio"}:
        return red
    if source == "bump_front_ok.003" or source == "lamp_01":
        return chrome
    if source in {"bump_front_ok","boot_ok"} and slot == "bump_front_ok.2":
        return gold
    if source == "bump_front_ok" and slot == "escape.0" and max(p.x for p in points)> .49:
        return chrome
    if source == "boot_ok":
        return chrome
    if source == "Fender_lig":
        return amber
    if source == "escape":
        return steel
    if label == "24 Salon":
        return fabric
    return dark


def surface_name(source, material, label):
    if source.startswith("roda"):
        return {alloy:"Disk va kegaylar", rubber:"Shina", steel:"Tormoz diski",
                gold:"Disk emblemasi", dark:"Boltlar va markaz"}[material]
    if source == "fara":
        return {lens:"Fara linzasi", chrome:"Fara reflektori", optic_black:"Fara korpusi"}[material]
    if source.startswith("door_") and material == window:
        return "Eshik oynasi"
    return label


def copy_mesh(name, src, points, faces, material_ids, group, transform):
    source_mesh = src.data
    used = sorted({vertex for face in faces for vertex in face.vertices})
    remap = {old: new for new, old in enumerate(used)}
    vertices = [transform @ points[index] for index in used]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], [tuple(remap[i] for i in face.vertices) for face in faces])
    mesh.update()
    for material in dict.fromkeys(material_ids):
        mesh.materials.append(material)
    normals = []
    normal_matrix = (transform @ src.matrix_world).to_3x3().inverted().transposed()
    for new, old, material in zip(mesh.polygons, faces, material_ids):
        new.use_smooth = old.use_smooth
        new.material_index = mesh.materials.find(material.name)
        normals.extend((normal_matrix @ source_mesh.corner_normals[loop].vector).normalized()
                       for loop in old.loop_indices)
    if any(face.use_smooth for face in faces):
        mesh.normals_split_custom_set(normals)
    for layer in source_mesh.uv_layers:
        uv = mesh.uv_layers.new(name=layer.name)
        for new, old in zip(mesh.polygons, faces):
            for to, origin in zip(new.loop_indices, old.loop_indices):
                uv.uv[to].vector = layer.uv[origin].vector
    obj = bpy.data.objects.new(name, mesh)
    group.objects.link(obj)
    obj["source_mesh"] = src.name
    return obj


args = options()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
prior = next(c for c in bpy.data.collections
             if c.name in {"COBALT • Exterior","Cobalt • reference fitted"})
prior_collections = [prior]
for collection in prior_collections:
    prior_collections.extend(collection.children)
    collection.name = "Old • "+collection.name
original_objects = set(bpy.data.objects)
original_materials = set(bpy.data.materials)
bpy.ops.import_scene.fbx(filepath=str(args.source.resolve()), use_image_search=False)
bpy.context.view_layer.update()
imported = [obj for obj in bpy.data.objects if obj not in original_objects]
source_objects = {obj.name: obj for obj in imported}

paint = bpy.data.materials["Champagne • clearcoat"]
window = bpy.data.materials["Automotive solar glass"]
lens = bpy.data.materials["Clear polycarbonate"]
dark = bpy.data.materials["EPDM and inner plastic"]
optic_black = finish("Dark headlamp optic shell", (.039,.046,.054), .12, .36, .16)
rubber = bpy.data.materials["Tire sidewall"]
alloy = bpy.data.materials["Machined alloy"]
chrome = bpy.data.materials["Polished chrome"]
silver_light = bpy.data.materials["Reflective lens optic"]
steel = bpy.data.materials["Cast steel"]
gold = bpy.data.materials["Gold emblem"]
red = finish("Tail light red molded reflector", (.55,.013,.024), .22, .28, .35)
amber = finish("Indicator amber reflector", (.77,.26,.037), .17, .30, .38)
fabric = finish("Cabin graphite fabric", (.049,.053,.057), 0, .77)

wheel_names = {"roda.002","roda.004","roda.008","roda.012"}
skipped = {name for name in source_objects if ".ColMesh" in name
           or (name.startswith("roda") and name not in wheel_names)}
plate_donors = ("bump_front_ok.005", "bump_rear_ok.001")
measured = {}
for name in ("bump_front_ok.002","bump_rear_ok",*wheel_names,*plate_donors):
    obj=source_objects[name]
    measured[name] = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
nose = max(p.y for p in measured["bump_front_ok.002"])
tail = min(p.y for p in measured["bump_rear_ok"])
floor = min(p.z for name in wheel_names for p in measured[name])
scale = 4.479 / (nose - tail)
transform = (Matrix.Translation(Vector(((nose + tail)/2*scale, 0, -floor*scale)))
             @ Matrix.Scale(scale,4) @ Matrix.Rotation(math.pi/2,4,"Z"))

car = new_collection("Cobalt • reference fitted", scene.collection)
parts = new_collection("Qismlar • editable assemblies", car)
groups = {}
origins = {}
for src in sorted(imported,key=lambda obj:obj.name):
    if src.type != "MESH" or src.name in skipped or src.name in plate_donors:
        continue
    points = [src.matrix_world @ vert.co for vert in src.data.vertices]
    batches = defaultdict(list)
    if src.name.startswith("roda"):
        islands = [list(src.data.polygons)]
    else:
        islands = face_islands(src.data)
    for island in islands:
        coordinates = [points[i] for i in {i for poly in island for i in poly.vertices}]
        centre = centre_of(coordinates)
        source_slots = {src.material_slots[face.material_index].name.lower() for face in island}
        group, name = assembly(src.name, centre, coordinates, source_slots)
        mats = [material_for(src.name,src.material_slots[p.material_index].name.lower(),
                             p.material_index,centre,coordinates,group) for p in island]
        if group.startswith("18 "):
            rocker, quarter = [], []
            for face, material in zip(island,mats):
                v = src.matrix_world @ face.center
                below_sill = -.49 > v.z and -1.30 < v.y < 1.06
                (rocker if below_sill else quarter).append((face,material))
            if rocker:
                side = "chap" if centre.x < 0 else "o‘ng"
                batches[shell_group("10 Ostona",side),"Ostona"].extend(rocker)
                if not quarter:
                    continue
                island,mats = map(list,zip(*quarter))
        for face, material in zip(island,mats):
            label = surface_name(src.name,material,name)
            batches[group,label].append((face,material))
    for (group,name), face_materials in batches.items():
        if group not in groups:
            groups[group] = new_collection(group, parts)
        faces,mats = map(list,zip(*face_materials))
        copy_mesh(name,src,points,faces,mats,groups[group],transform)
        if src.parent and src.parent.name.endswith("_dummy") and src.name in {
            "bonnet_ok","boot_ok","door_lf_ok","door_rf_ok","door_lr_ok.001","door_rr_ok.001"
        } and not group.startswith("11 "):
            origins[group] = transform @ src.parent.matrix_world.translation
        elif src.name in wheel_names:
            origins[group] = transform @ centre_of(measured[src.name])

# Reuse the previous editable plate, frame, edge, and lettering on the donor's mounts.
replacement_names = []
for index, (donor, target) in enumerate(zip(plate_donors, ("02 Old bamper","16 Orqa bamper"))):
    points = [transform @ v for v in measured[donor]]
    centre = centre_of(points)
    direction = -1 if index==0 else 1
    face_x = (min if index==0 else max)(p.x for p in points)
    suffix = ".001" if index else ""
    plate = bpy.data.objects["License plate"+suffix]
    bpy.context.view_layer.update()
    plate_centre = plate.matrix_world.translation
    new_width = max(p.y for p in points)-min(p.y for p in points)
    width_scale = new_width / plate.dimensions.y
    for name in ("License plate frame","License plate bright edge","License plate","COBALT"):
        source = bpy.data.objects[name+suffix]
        obj = source.copy()
        obj.data = source.data.copy()
        replacement_names.append((obj,source.name))
        groups[target].objects.link(obj)
        obj.parent = None
        world = source.matrix_world.copy()
        anchor = Vector((face_x+direction*.003, centre.y, centre.z))
        change = Matrix.Diagonal((1.0,width_scale,.92,1.0))
        obj.matrix_world = Matrix.Translation(anchor) @ change @ Matrix.Translation(-plate_centre) @ world

for obj in list(prior.all_objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for collection in reversed(prior_collections):
    bpy.data.collections.remove(collection)
for obj,name in replacement_names:
    obj.name = name
source_meshes = {obj.data for obj in imported if obj.type=="MESH"}
for obj in imported:
    bpy.data.objects.remove(obj, do_unlink=True)
for mesh in source_meshes:
    if mesh.users==0:
        bpy.data.meshes.remove(mesh)
for material in list(bpy.data.materials):
    if material not in original_materials and material.users==0:
        bpy.data.materials.remove(material)
for image in list(bpy.data.images):
    if image.users==0 and image.source=="FILE" and not image.packed_file:
        bpy.data.images.remove(image)

datum = bpy.data.objects.new("Cobalt vehicle datum",None)
car.objects.link(datum)
datum.empty_display_size=.22
datum["body_length_m"] = 4.479
datum["source_uniform_scale"] = scale
datum["reference"] = args.source.name
bpy.context.view_layer.update()
for label, group in groups.items():
    objects = list(group.objects)
    pivot = bpy.data.objects.new(label.split(" ",1)[1],None)
    group.objects.link(pivot)
    pivot.parent = datum
    centre = origins.get(label)
    if centre is None:
        centre = centre_of([obj.matrix_world @ Vector(v) for obj in objects for v in obj.bound_box])
    pivot.location = centre
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.empty_display_size = .12
    pivot["assembly"] = label
    bpy.context.view_layer.update()
    for obj in objects:
        world = obj.matrix_world.copy()
        obj.parent = pivot
        obj.matrix_world = world

scene["Source"] = "User-supplied cobalt.fbx • original body seams and optics geometry"
scene["Scale"] = "Uniformly fit the 4.479 m bumper-to-bumper datum."
scene["Finishes"] = "Rebuilt Blender paint, glass, optics and wheels. External FBX images were absent."
studio = next(c for c in bpy.data.collections if c.name.startswith("Studio"))
floor_mat = bpy.data.objects["Studio seamless floor"].data.materials[0]
cove = bpy.data.objects["Cyclorama • curved neutral wall"]
elevation = bpy.data.materials.get("Cove • neutral horizon", floor_mat)
cameras = {obj.name[:2]:obj for obj in studio.objects if obj.type=="CAMERA"}
scene.camera = cameras["01"]
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.cycles.samples = 24 if args.draft else 64
scene.cycles.adaptive_threshold = .048 if args.draft else .018
scene.render.resolution_y = 680 if args.draft else 1010
scene.render.resolution_x = 1060 if args.draft else 1560
scene.render.filepath = "//Cobalt_front.png"
if not args.draft:
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "Chevrolet_Cobalt.blend"))
for prefix,label in (("01","front"),("02","rear"),("04","straight"),("03","side"),("06","back"),("05","top")):
    scene.camera = cameras[prefix]
    ortho_face = prefix in {"04","06"}
    cove.data.materials[0] = elevation if ortho_face or prefix=="03" else floor_mat
    scene.render.resolution_x = scene.render.resolution_y if ortho_face else (1060 if args.draft else 1560)
    scene.render.filepath = str(output / f"Cobalt_{label}.png")
    bpy.ops.render.render(write_still=True)
