# No Truce With The Furies
import pythia8
import matplotlib.pyplot as plt

# Proton
pythia = pythia8.Pythia()
pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:eA = 4000")

# Nitrogen - 14
pythia.readString("Beams:idB = 1000070140")
pythia.readString("Beams:eB = 1400")
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

# Data
e_num = 0
ph_num = 0
n_num = 0
pr_num = 0
hi_num = 0
energies = []
par_dict = {}
event = pythia.event
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

final_state_particles = [prt for prt in pythia.event if prt.status() > 0]
print(f"there are {len(final_state_particles)} final-state particles")
print("energy of the 1st collision product:")
p = pythia.event[3]
print(f"{p.e():.2f} GeV")
print("Identity of the 1st collision product:")
print(p.id())
print("Energies:", energies)
print(par_dict)

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
plt.show()
