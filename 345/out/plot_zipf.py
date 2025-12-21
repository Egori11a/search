import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("zipf.csv")
plt.figure()
plt.loglog(df["rank"], df["freq"], marker=".", linestyle="None")
plt.loglog(df["rank"], df["zipf"])
plt.grid(True, which="both")
plt.title("Zipf law (log-log)")
plt.xlabel("rank")
plt.ylabel("frequency")
plt.savefig("zipf.png", dpi=160)
print("saved zipf.png")
