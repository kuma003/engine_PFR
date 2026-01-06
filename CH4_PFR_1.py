import cantera as ct
import csv

# ============================
# 1. Simulation parameters
# ============================
p = ct.one_atm          # 圧力 [Pa]
Tin = 1500.0            # 入口温度 [K]
comp = 'CH4:1, O2:1, AR:0.5'  # 入口組成（モル比）

vin = 30                # 入口流速 [m/s]
length = 0.36           # 反応器全長 [m]
area = 9.07e-4           # 断面積 [m^2]
n_reactor = 1000         # 分割セル数

# ============================
# 2. Gas & mass flow settings
# ============================
gas = ct.Solution('gri30.yaml')   # GRI3.0 機構
gas.TPX = Tin, p, comp

# 質量流量 mdot = ρ * u * A
mdot = vin * area * gas.density

# セル長さ
dx = length / n_reactor

# ============================
# 3. 1 セル目の反応器系を構築
# ============================
# 反応器（理想気体・定容）
r = ct.IdealGasReactor(gas)
r.volume = area * dx

# 上流・下流のリザーバ
upstream = ct.Reservoir(gas, name='upstream')
downstream = ct.Reservoir(gas, name='downstream')

# 上流 → 反応器：質量流量一定
mfc = ct.MassFlowController(upstream, r, mdot=mdot)

# 反応器 → 下流：圧力制御（K は適当でよい）
pc = ct.PressureController(r, downstream, primary=mfc, K=1.0e-5)

# 時間発展ソルバ
sim = ct.ReactorNet([r])

# ============================
# 4. 出力ファイルの準備
# ============================
outfile = open('pfr_CH4_O2_Ar.csv', 'w', newline='')
writer = csv.writer(outfile)

header = ['Distance (m)', 'u (m/s)', 'res_time (s)',
          'T (K)', 'P (Pa)'] + gas.species_names
writer.writerow(header)

# ============================
# 5. 擬似 PFR 計算ループ
# ============================
t_res = 0.0  # トータル滞留時間

for n in range(n_reactor):
    # このセルの初期状態を upstream にコピー
    gas.TDY = r.thermo.TDY
    upstream.syncState()

    # 反応器ネットを再初期化 → 定常まで計算
    sim.reinitialize()
    sim.advance_to_steady_state()

    # 現在セル位置
    dist = (n + 1) * dx

    # 流速 u = mdot / (A * ρ)
    u = mdot / (area * r.thermo.density)

    # セル内質量 / mdot = このセルでの滞在時間
    t_res += r.mass / mdot

    # 結果書き出し
    row = [dist, u, t_res, r.T, r.thermo.P] + list(r.thermo.X)
    writer.writerow(row)

outfile.close()

print("Finished. Results saved to 'pfr_CH4_O2_Ar.csv'")