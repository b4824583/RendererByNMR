from src.utils import mesh
mean_shape = mesh.fetch_mean_shape("cachedir/template_shape/bird_template_orbifold.obj", mean_centre_vertices=True)
verts = mean_shape['verts']
faces = mean_shape['faces']
verts_uv = mean_shape['verts_uv']
faces_uv = mean_shape['faces_uv']
print("faces uv max:" + str(faces_uv.max()))
print("faces uv min:" + str(faces_uv.min()))
faces_uv = ((faces_uv + 2.1547) / 3.1547*2)-1 + 1e-12
# faces_uv = 2*faces_uv-1 # Normalize to [-1,1]
print("faces uv max by normal:" + str(faces_uv.max()))
print("faces uv min by normal:" + str(faces_uv.min()))