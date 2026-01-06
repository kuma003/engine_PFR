import cantera as ct
import csv

# ============================
# 1. Simulation parameters
# ============================
p = ct.one_atm
Tin = 1500.0

# 一段目の混合気
comp_main = 'CH4:1, O2:3, AR:0.5'

# 二段目で追加する燃料（純メタン）
comp_fuel = 'CH4:1'

# 質量流量
mdot_main = 0.01
mdot_CH4_add = 0.001

length = 0.36
area = 9.07e-4
n_reactor = 1000
dx = length / n_reactor

# 注入位置
x_inj = 0.18
inj_cell = int(x_inj / dx)

# ============================
# 2. Gas objects
# ============================
gas = ct.Solution('gri30.yaml')
gas.TPX = Tin, p, comp_main

gas_fuel = ct.Solution('gri30.yaml')
gas_fuel.TPX = Tin, p, comp_fuel

# ============================
# 3. Reactor setup
# ============================
r = ct.IdealGasReactor(gas)
r.volume = area * dx

upstream = ct.Reservoir(gas)
downstream = ct.Reservoir(gas)

# 最初は主流のみ
mdot_total = mdot_main
mfc = ct.MassFlowController(upstream, r, mdot=mdot_total)
pc = ct.PressureController(r, downstream, primary=mfc, K=1e-5)

sim = ct.ReactorNet([r])

# ============================
# 4. Output
# ============================
outfile = open('pfr_two_stage_CH4.csv', 'w', newline='')
writer = csv.writer(outfile)
writer.writerow(['Distance (m)', 'u (m/s)', 'res_time (s)', 'T (K)', 'P (Pa)'] + gas.species_names)

# ============================
# 5. PFR marching
# ============================
t_res = 0.0
injection_done = False

for n in range(n_reactor):

    # ---- 二段目の混合イベント ----
    if (not injection_done) and (n >= inj_cell):
        injection_done = True

        # 一段目出口ガス
        gas_main = ct.Solution('gri30.yaml')
        gas_main.TDY = r.thermo.TDY

        # 質量分率混合
        Y_mix = (mdot_main * gas_main.Y + mdot_CH4_add * gas_fuel.Y) / (mdot_main + mdot_CH4_add)

        # 比エンタルピ混合
        h_mix = (mdot_main * gas_main.enthalpy_mass +
                 mdot_CH4_add * gas_fuel.enthalpy_mass) / (mdot_main + mdot_CH4_add)

        # 混合状態をセット（HPY を使う）
        gas.HPY = h_mix, p, Y_mix
        r.syncState()

        # 合計流量に更新
        mdot_total = mdot_main + mdot_CH4_add
        mfc = ct.MassFlowController(upstream, r, mdot=mdot_total)
        pc = ct.PressureController(r, downstream, primary=mfc, K=1e-5)
        sim = ct.ReactorNet([r])

    # ---- セル入口状態を upstream にコピー ----
    gas.TDY = r.thermo.TDY
    upstream.syncState()

    # ---- 定常まで解く ----
    sim.reinitialize()
    sim.advance_to_steady_state()

    # ---- 出力 ----
    dist = (n + 1) * dx
    rho = r.thermo.density
    u = mdot_total / (area * rho)
    t_res += r.mass / mdot_total

    writer.writerow([dist, u, t_res, r.T, r.thermo.P] + list(r.thermo.X))

outfile.close()
print("Finished.")