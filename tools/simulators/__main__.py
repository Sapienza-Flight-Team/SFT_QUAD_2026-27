import os

print("Selezionare il simulatore desiderato tra i seguenti:")
# cerco nella path corrente tutti i file .py che iniziano per sim
elems = [
    x
    for x in (os.listdir(os.path.dirname(__file__)))
    if x.startswith("sim") and x.endswith(".py")
]
for i, elem in enumerate(elems):
    print(f"{(i+1)}) {elem}")
# seleziono il simulatore
idx = int(input("Inserisci il numero del simulatore: "))
print("simulatore selezionato:", elems[idx - 1])
__import__(f"tools.simulators.{elems[idx-1][:-3]}")
