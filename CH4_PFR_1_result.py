import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================
# 1. CSV 読み込み
# ============================
df = pd.read_csv("pfr_CH4_O2_Ar.csv")

x = df["Distance (m)"]
T = df["T (K)"]
u = df["u (m/s)"]

# 主要種（必要に応じて追加）
species = ["CH4", "O2", "CO", "CO2", "H2O", "H2", "OH", "O", "AR"]

# ============================
# 2. 温度プロファイル
# ============================
plt.figure(figsize=(7,4))
plt.plot(x, T, linewidth=2)
plt.xlabel("Distance [m]")
plt.ylabel("Temperature [K]")
plt.title("Temperature Profile along PFR")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 3. 流速プロファイル
# ============================
plt.figure(figsize=(7,4))
plt.plot(x, u, linewidth=2)
plt.xlabel("Distance [m]")
plt.ylabel("Velocity [m/s]")
plt.title("Velocity Profile along PFR")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 4. 主要化学種のモル分率
# ============================
plt.figure(figsize=(8,5))

for sp in species:
    if sp in df.columns:
        plt.plot(x, df[sp], label=sp)

plt.xlabel("Distance [m]")
plt.ylabel("Mole Fraction")
plt.title("Species Profiles along PFR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()