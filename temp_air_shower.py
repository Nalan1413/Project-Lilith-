# No Truce With The Furies
import pythia8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
pythia.readString("PartonLevel:MPI = on")
pythia.readString("MultipartonInteractions:alphaSvalue = 0.130")
pythia.readString("MultipartonInteractions:pTmaxMatch = 0")
pythia.readString("MultipartonInteractions:bProfile = 1")
pythia.readString("MultipartonInteractions:allowRescatter = on")

pythia.init()
event = pythia.event

for events in range(1, 51):
    pythia.next()
    print(f"Number of particles {event.size()}")
    if events % 5 == 0:
        collision_count = 0
        for particle in event:

            # Debug
            if particle.isFinal() != 10:
                if particle.id() == 1000070140:
                    collision_count += 1
                    print(f"event {events} number of N14: {collision_count}")

            # Add particles and check for collision
            if particle.id() == 2212 and particle.isFinal():
                print("found")
                # Particle::Particle(int id, int status = 0, int mother1 = 0, int mother2 = 0, int daughter1 = 0, int daughter2 = 0,
                # int col = 0, int acol = 0, double px = 0., double py = 0., double pz = 0., double e = 0., double m = 0., double scale = 0., double pol = 9.)
                n14 = pythia8.Particle(1000070140, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 13144.859)
                n14.vProd(particle.xProd(), particle.yProd(), particle.zProd(), particle.tProd())
                event.append(n14)
                print(n14.xProd(),n14.yProd(),n14.zProd())

# Init data and Graphs
shower_particles = [particle for particle in event]
particle_x = [particle.xProd() for particle in shower_particles]
particle_y = [particle.yProd() for particle in shower_particles]
particle_z = [particle.zProd() for particle in shower_particles]
particle_en = [particle.e() for particle in shower_particles]

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


parent_ids_m1 = [particle.mother1() for particle in shower_particles]
parent_ids_m2 = [particle.mother2() for particle in shower_particles]
particle_eta = [particle.eta() for particle in shower_particles]
particle_phi = [particle.phi() for particle in shower_particles]

pythia.stat()

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
    ax_polar.scatter(particle_phi[i], particle_eta[i], c=particle_en[i], cmap='viridis', s=50, alpha=0.7)
    if parent_ids_m1[i] != 0:
        ax_polar.plot([event[parent_ids_m1[i]].phi(), particle_phi[i]],
                      [event[parent_ids_m1[i]].eta(), particle_eta[i]], color='gray', linestyle='--')
    if parent_ids_m2[i] != 0:
        ax_polar.plot([event[parent_ids_m2[i]].phi(), particle_phi[i]],
                      [event[parent_ids_m2[i]].eta(), particle_eta[i]], color='gray', linestyle='--')

ax_polar.set_title('Polar Coordinates')
cbar_polar = plt.colorbar(ax_polar.scatter([], [], c=[], cmap='viridis', s=50, alpha=0.7), ax=ax_polar, orientation='vertical')
cbar_polar.set_label('Energy')

fig = plt.figure(figsize=(10, 8))
ax_3d = fig.add_subplot(111, projection='3d')

sc = ax_3d.scatter(particle_x, particle_y, particle_z, c=particle_en, cmap='viridis', s=50, alpha=0.7)

for i in range(len(shower_particles)):
    if parent_ids_m1[i] != 0:
        ax_3d.plot([particle_x[parent_ids_m1[i]], particle_x[i]],
                   [particle_y[parent_ids_m1[i]], particle_y[i]],
                   [particle_z[parent_ids_m1[i]], particle_z[i]], color='gray', linestyle='--')
    if parent_ids_m2[i] != 0:
        ax_3d.plot([particle_x[parent_ids_m2[i]], particle_x[i]],
                   [particle_y[parent_ids_m2[i]], particle_y[i]],
                   [particle_z[parent_ids_m2[i]], particle_z[i]], color='gray', linestyle='--')

ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('3D Coordinates')

ax_3d.set_xlim(-25, 25)
ax_3d.set_ylim(-25, 25)
ax_3d.set_zlim(-25, 25)


cbar_ax_3d = plt.colorbar(sc, ax=ax_3d, orientation='vertical')
cbar_ax_3d.set_label('Energy')
plt.show()
num_points = len(sc.get_offsets())
print(f"Number of points: {num_points}")
