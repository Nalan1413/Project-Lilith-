# No Truce With The Furies
import pythia8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SecondaryCascade import Cascade
import pickle

# Proton
pythia = pythia8.Pythia()
pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:eA = 2600")

# Nitrogen - 14
pythia.readString("Beams:idB = 1000070140")
pythia.readString("Beams:eB = 14")

pythia.readString("Beams:frameType = 3")
pythia.readString("HeavyIon:SigFitNGen = 0")
pythia.readString("HeavyIon:SigFitDefPar = 29.95,2.19,0.60")
pythia.readString("HardQCD:all = on")

"""
pythia.readString("PartonLevel:MPI = on")
pythia.readString("MultipartonInteractions:alphaSvalue = 0.130")
pythia.readString("MultipartonInteractions:pTmaxMatch = 0")
pythia.readString("MultipartonInteractions:bProfile = 1")
pythia.readString("MultipartonInteractions:allowRescatter = on")
"""

pythia.init()
for events in range(5):
    pythia.next()
    event = pythia.event
    print(f"目前数量{event.size()}")
    count = 0
    sec = []
    for idx, particle in enumerate(event):
        # Add particles and check for collision
        if particle.id() == 2212 and particle.isFinal():
            print("found")
            Cas = Cascade(particle.e(), particle.px(), particle.py(), particle.pz(),
                          particle.xProd(), particle.yProd(), particle.zProd(), idx)
            sec.append(Cas.transformation())
            count += 1

        if count > 5:
            break



# Data and Graphs
# Primary shower
shower_particles = [particle for particle in event]
particle_x = [particle.xProd() for particle in shower_particles]
particle_y = [particle.yProd() for particle in shower_particles]
particle_z = [particle.zProd() for particle in shower_particles]
particle_en = [particle.e() for particle in shower_particles]

parent_ids_m1 = [particle.mother1() for particle in shower_particles]
parent_ids_m2 = [particle.mother2() for particle in shower_particles]
particle_eta = [particle.eta() for particle in shower_particles]
particle_phi = [particle.phi() for particle in shower_particles]


# Secondary Shower
sec_particle_x = []
sec_particle_y = []
sec_particle_z = []
sec_particle_en = []
sec_parent_ids_m1 = []
sec_parent_ids_m2 = []
sec_particle_eta = []
sec_particle_phi = []
for m in range(len(sec)):
    sec_particle_x.append([particle.xProd() for particle in sec[m]])
    sec_particle_y.append([particle.yProd() for particle in sec[m]])
    sec_particle_z.append([particle.zProd() for particle in sec[m]])
    sec_particle_en.append([particle.e() for particle in sec[m]])

    sec_parent_ids_m1.append([particle.mother1() for particle in sec[m]])
    sec_parent_ids_m2.append([particle.mother2() for particle in sec[m]])
    sec_particle_eta.append([particle.eta() for particle in sec[m]])
    sec_particle_phi.append([particle.phi() for particle in sec[m]])


# Energy and names
energies = []
par_dict = {}
for i in range(event.size()):
    particle = event[i]
    particle_name = event[i].name()

    if particle.isFinal():
        energies.append(particle.e())

    if particle_name in par_dict:
        par_dict[particle_name] += 1
    else:
        par_dict[particle_name] = 1

shower_color = 'blue'
sec_color = 'red'
shower_colors = [shower_color] * len(particle_x)
sec_colors = [sec_color] * sum(len(particles) for particles in sec_particle_x)

plt.figure(figsize=(10, 10))
plt.hist(energies, bins=50, range=(0, 200), density=True, alpha=0.7, color='blue')
plt.xlabel('Particle Energy (GeV)')
plt.ylabel('Normalized Frequency')
plt.title('Particle Energy Distribution in Pythia8 Simulation')

labels = list(par_dict.keys())
values = list(par_dict.values())
plt.figure(figsize=(10, 4))
plt.bar(labels, values, color='skyblue')
plt.xlabel('Particle Name')
plt.ylabel('Frequency')
plt.title('Particle ID Distribution')

fig, ax_polar = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(10, 8))
for i in range(len(shower_particles)):
    ax_polar.scatter(particle_phi[i], particle_eta[i], c=shower_colors[i], cmap='viridis', s=50, alpha=0.7)
    if parent_ids_m1[i] != 0:
        ax_polar.plot([event[parent_ids_m1[i]].phi(), particle_phi[i]],
                      [event[parent_ids_m1[i]].eta(), particle_eta[i]], color='gray', linestyle='--')
    if parent_ids_m2[i] != 0:
        ax_polar.plot([event[parent_ids_m2[i]].phi(), particle_phi[i]],
                      [event[parent_ids_m2[i]].eta(), particle_eta[i]], color='gray', linestyle='--')

for j in range(len(sec)):
    for k in range(sec[j].size()):
        ax_polar.scatter(sec_particle_phi[j][k], sec_particle_eta[j][k], c=sec_colors[j], cmap='viridis', s=50, alpha=0.7)
        if sec_parent_ids_m1[j][k] != 0:
            ax_polar.plot([sec_particle_phi[j][sec_parent_ids_m1[j][k]], sec_particle_phi[j][k]],
                          [sec_particle_eta[j][sec_parent_ids_m1[j][k]], sec_particle_eta[j][k]], color='gray', linestyle='--')
        if sec_parent_ids_m2[j][k] != 0:
            ax_polar.plot([sec_particle_phi[j][sec_parent_ids_m2[j][k]], sec_particle_phi[j][k]],
                          [sec_particle_eta[j][sec_parent_ids_m2[j][k]], sec_particle_eta[j][k]], color='gray', linestyle='--')

ax_polar.set_title('Polar Coordinates')
cbar_polar = plt.colorbar(ax_polar.collections[0], ax=ax_polar, orientation='vertical')
cbar_polar.set_label('Energy')

fig = plt.figure(figsize=(10, 8))
ax_3d = fig.add_subplot(111, projection='3d')

ax_3d.scatter(particle_x, particle_y, particle_z, c=shower_colors, cmap='viridis', s=50, alpha=0.7)

for i in range(len(shower_particles)):
    if parent_ids_m1[i] != 0:
        ax_3d.plot([event[parent_ids_m1[i]].xProd(), particle_x[i]],
                   [event[parent_ids_m1[i]].yProd(), particle_y[i]],
                   [event[parent_ids_m1[i]].zProd(), particle_z[i]], color='gray', linestyle='--')
    if parent_ids_m2[i] != 0:
        ax_3d.plot([event[parent_ids_m2[i]].xProd(), particle_x[i]],
                   [event[parent_ids_m2[i]].yProd(), particle_y[i]],
                   [event[parent_ids_m2[i]].zProd(), particle_z[i]], color='gray', linestyle='--')

for j in range(len(sec)):
    ax_3d.scatter(sec_particle_x[j], sec_particle_y[j], sec_particle_z[j], c=sec_colors[j],
                  cmap='viridis', s=50, alpha=0.7)

    for k in range(sec[j].size()):
        if sec_parent_ids_m1[j][k] != 0:
            ax_3d.plot([sec_particle_x[j][sec_parent_ids_m1[j][k]], sec_particle_x[j][k]],
                       [sec_particle_y[j][sec_parent_ids_m1[j][k]], sec_particle_y[j][k]],
                       [sec_particle_z[j][sec_parent_ids_m1[j][k]], sec_particle_z[j][k]], color='gray', linestyle='--')
        if sec_parent_ids_m2[j][k] != 0:
            ax_3d.plot([sec_particle_x[j][sec_parent_ids_m2[j][k]], sec_particle_x[j][k]],
                       [sec_particle_y[j][sec_parent_ids_m2[j][k]], sec_particle_y[j][k]],
                       [sec_particle_z[j][sec_parent_ids_m2[j][k]], sec_particle_z[j][k]], color='gray', linestyle='--')

ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('3D Coordinates')

ax_3d.set_xlim(-0.002, 0.002)
ax_3d.set_ylim(-0.002, 0.002)
ax_3d.set_zlim(-0.002, 0.002)

cbar_ax_3d = plt.colorbar(ax_3d.collections[0], ax=ax_3d, orientation='vertical')
cbar_ax_3d.set_label('Energy')

print(f"Number of primary shower particles{len(particle_x)}")
num_points = len(particle_x) + sum([len(particles) for particles in sec_particle_x])
print(f"Number of shower particles: {num_points}")

plt.show()

Primary_shower_data = {'particle_x': particle_x, 'particle_y': particle_y, 'particle_z': particle_z, 'particle_en': particle_en,
                       'parent_ids_m1': parent_ids_m1, 'parent_ids_m2': parent_ids_m2, 'particle_eta': particle_eta, 'particle_phi': particle_phi}
Secondary_shower_data = {'sec_particle_x': sec_particle_x, 'sec_particle_y': sec_particle_y, 'sec_particle_z': sec_particle_z, 'sec_particle_en': sec_particle_en,
                         'sec_parent_ids_m1': sec_parent_ids_m1, 'sec_parent_ids_m2': sec_parent_ids_m2, 'sec_particle_eta': sec_particle_eta, 'sec_particle_phi': sec_particle_phi}
with open('data.pkl', 'wb') as f:
    pickle.dump(Primary_shower_data, f)
    pickle.dump(Secondary_shower_data, f)
