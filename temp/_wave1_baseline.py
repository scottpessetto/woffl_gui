"""Recompute E-41 batch reference values with the fixed solver and verify
that each solved psu actually drives the discharge residual to ~zero."""

import woffl.assembly.solopump as so
from woffl.assembly.batchpump import BatchPump
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

surf_pres = 210
ppf_surf = 3168
tsu = 80

tube = Pipe(out_dia=4.5, thick=0.5)
case = Pipe(out_dia=6.875, thick=0.5)
wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)
e41_ipr = InFlow(qwf=246, pwf=1049, pres=1400)
e41_res = ResMix(0.894, 600, BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())
e41_profile = WellProfile.schrader()

jp_list = BatchPump.jetpump_list(
    ["9", "10", "11", "12", "13", "14", "15", "16"], ["X", "A", "B", "C", "D", "E"]
)
batch = BatchPump(
    surf_pres, tsu, ppf_surf, wellbore, e41_profile, e41_ipr, e41_res,
    FormWater.schrader(), jpump_direction="reverse", wellname="MPE-41",
)
df = batch.batch_run(jp_list)

print("rows:", len(df))
print("errors:", df[df["error"] != "na"][["nozzle", "throat", "error"]].to_string())
print("sonic count:", df["sonic_status"].sum())

for noz, thr in [("9", "X"), ("9", "D"), ("12", "B"), ("16", "E")]:
    row = df[(df["nozzle"] == noz) & (df["throat"] == thr)].iloc[0]
    print(
        f"{noz}{thr}: qoil={row['qoil_std']:.2f} totl_wat={row['totl_wat']:.2f} "
        f"mach_te={row['mach_te']:.4f} psu={row['psu_solv']:.2f} sonic={row['sonic_status']}"
    )
    # verify the residual at the solved psu for non-sonic rows
    if not row["sonic_status"]:
        res, qoil, fwat, qnz, mach = so.discharge_residual(
            row["psu_solv"], surf_pres, tsu, ppf_surf,
            JetPump(noz, thr, 0.01, 0.03, 0.3, 0.4),  # match jetpump_list coefs
            wellbore, e41_profile, e41_ipr, e41_res,
            FormWater.schrader(), "reverse",
        )
        print(f"    residual at solved psu: {res:+.2f} psid")
