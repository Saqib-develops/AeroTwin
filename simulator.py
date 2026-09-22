import numpy as np
import pandas as pd

def mission_profile(n=180):
    t=np.arange(n); phase=[]; throttle=[]; altitude=[]; ambient=[]
    for i in t:
        if i<25: ph,th,alt='TAKEOFF',.85,150+i*90
        elif i<65: ph,th,alt='CLIMB',.78,2400+(i-25)*180
        elif i<145: ph,th,alt='CRUISE',.62,9600
        elif i<165: ph,th,alt='HIGH LOAD',.88,10500
        else: ph,th,alt='DESCENT',.48,max(800,10500-(i-165)*450)
        phase.append(ph); throttle.append(th); altitude.append(alt); ambient.append(22+.0015*alt)
    return pd.DataFrame({'t':t,'mission_phase':phase,'throttle':throttle,'altitude_ft':altitude,'ambient_temp':ambient})

def simulate(scenario='Normal', n=180, seed=42, throttle=None, altitude=None, ambient=None):
    rng=np.random.default_rng(seed); m=mission_profile(n)
    if throttle is not None: m['throttle']=throttle
    if altitude is not None: m['altitude_ft']=altitude
    if ambient is not None: m['ambient_temp']=ambient
    t=m.t.to_numpy(); th=m.throttle.to_numpy(); alt=m.altitude_ft.to_numpy(); amb=m.ambient_temp.to_numpy()
    af=np.clip(alt/10000,0,1.5); load=.35+.65*th
    rpm=3300+1150*th-45*af+rng.normal(0,22,n)
    fuel=7.5+15.5*th+.7*af+rng.normal(0,.35,n)
    egt=500+190*th+18*af+.45*(amb-25)+rng.normal(0,9,n)
    cht=135+52*th+10*af+.35*(amb-25)+rng.normal(0,2.8,n)
    oil_t=65+26*load+.25*(amb-25)+rng.normal(0,1.5,n)
    oil_p=66-10*load-2.5*af+rng.normal(0,1.3,n)
    vib=.10+.045*load+rng.normal(0,.007,n)
    batt=24.4-.35*th+rng.normal(0,.06,n)
    alt_h=np.clip(.98-.02*th+rng.normal(0,.004,n),0,1)
    inj=18+rng.normal(0,.12,n)
    fs=int(n*.58); ramp=np.clip((t-fs)/max(1,n-fs),0,1)
    if scenario=='Overheating / Thermal Degradation': cht+=42*ramp; egt+=82*ramp; oil_t+=15*ramp; vib+=.025*ramp
    elif scenario=='Lubrication Degradation': oil_p-=22*ramp; oil_t+=20*ramp; vib+=.075*ramp; cht+=7*ramp
    elif scenario=='Injector / Fuel Abnormality': inj+=2.2*ramp; fuel+=4.5*ramp; egt+=45*ramp; rpm-=180*ramp; vib+=.055*ramp
    elif scenario=='Mechanical / Vibration': vib+=.20*ramp; rpm+=75*np.sin(t/2.5)*ramp; oil_p-=6*ramp
    elif scenario=='Sensor Drift': cht+=24*ramp; egt-=18*ramp
    m=m.copy();
    for k,v in {'rpm':rpm,'cht':cht,'egt':egt,'oil_pressure':oil_p,'oil_temp':oil_t,'fuel_flow':fuel,'vibration':vib,'battery_voltage':batt,'alternator_health':alt_h,'injection_timing':inj}.items(): m[k]=v
    m['scenario']=scenario; m['fault_start']=fs
    return m

def expected(row):
    th=float(row.throttle); alt=float(row.altitude_ft); amb=float(row.ambient_temp); af=np.clip(alt/10000,0,1.5); load=.35+.65*th
    return {'rpm':3300+1150*th-45*af,'cht':135+52*th+10*af+.35*(amb-25),'egt':500+190*th+18*af+.45*(amb-25),'oil_pressure':66-10*load-2.5*af,'oil_temp':65+26*load+.25*(amb-25),'fuel_flow':7.5+15.5*th+.7*af,'vibration':.10+.045*load,'battery_voltage':24.4-.35*th,'alternator_health':.98-.02*th,'injection_timing':18.0}
