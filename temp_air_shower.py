# No Truce With The Furies
import pythia8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Proton
pythia = pythia8.Pythia()
pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:eA = 10000")

# Nitrogen - 14
pythia.readString("Beams:idB = 1000070140")
pythia.readString("Beams:eB = 14")
pythia.readString("Beams:frameType = 2")

# I am not sure what this part is supposed to do
pythia.readString("HeavyIon:SigFitErr = "
                  "0.02,0.02,0.1,0.05,0.05,0.0,0.1,0.0")
pythia.readString("HeavyIon:SigFitNGen = 20")
pythia.readString("HeavyIon:SigFitDefPar = 26.06,1.98,0.50")

# Set up
pythia.readString("HardQCD:all = on")
pythia.init()
pythia.next()
event = pythia.event

# Data
energies = []
par_dict = {}
for i in range(event.size()):

    particle = event[i]
    particle_name = event[i].name()

    if particle.isFinal() and particle.isParton() and pythia8.pythia.mayDecay(particle.id()):
        pythia8.pythia.forceTimeDecay(particle.id())

    elif particle.isFinal():
        energies.append(particle.e())

    elif particle_name in par_dict:
        par_dict[particle_name] += 1
    else:
        par_dict[particle_name] = 1

# print(f"    Particle {i + 1}: ID = {particle.id()}, PT = {particle.pT()}, Eta = {particle.eta()}, Phi = {particle.phi()}, Mass = {particle.m()}, Charge = {particle.charge()}")

# final_state_particles = [prt for prt in pythia.event if prt.status() > 0]
shower_particles = [particle for particle in event]

parent_ids_m1 = [particle.mother1() for particle in shower_particles]
parent_ids_m2 = [particle.mother2() for particle in shower_particles]
particle_eta = [particle.eta() for particle in shower_particles]
particle_phi = [particle.phi() for particle in shower_particles]
particle_en = [particle.e() for particle in shower_particles]
particle_x = [particle.xProd() for particle in shower_particles]
particle_y = [particle.yProd() for particle in shower_particles]
particle_z = [particle.zProd() for particle in shower_particles]

print("energy of the 1st collision product:")
p = pythia.event[3]
print(f"{p.e():.2f} GeV")
print("Identity of the 1st collision product:")
print(p.id())
print("Energies:", energies)
print(par_dict)
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

print("parent_ids_m1:", parent_ids_m1)
print("parent_ids_m2:", parent_ids_m2)
print(len(shower_particles))
print(len(parent_ids_m1))
print(len(parent_ids_m2))
print(particle_x)
print(particle_y)
print(particle_z)

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


ax_3d.set_xlim(-0.0000000000008, 0.0000000000008)
ax_3d.set_ylim(-0, 0.0000000000008)
ax_3d.set_zlim(-0.01, 0.01)

cbar_ax_3d = plt.colorbar(sc, ax=ax_3d, orientation='vertical')
cbar_ax_3d.set_label('Energy')
plt.show()

num_points = len(sc.get_offsets())
print(f"Number of points: {num_points}")
