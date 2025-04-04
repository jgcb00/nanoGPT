import os
from dataclasses import dataclass
from typing import List
import tyro
import pickle
import json
import tiktoken
from pathlib import Path
from arch.utils import get_model
import torch
from config import NanoConfig

from arch.lm import NanoLM

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

@dataclass
class Args:
    run_dir: Path  # something like logs/... (the dir that contains the .pt model)

    def __post_init__(self):
        assert self.run_dir.exists(), f"Run directory {self.run_dir} does not exist."

args = tyro.cli(Args)

# read config
with open(args.run_dir / 'config.pkl', 'rb') as f:
    config: NanoConfig = pickle.load(f)
config.rmsnorm = False
config.disable_scalable_softmax_for_local = False # False for loading old runs, True for newer ones

# define and load model, tokenizer
model = get_model(config)
model.cuda()

model_file = sorted(args.run_dir.glob("state_step*.pt"))[-1]
assert model_file.exists(), f"Model file {model_file} does not exist."

checkpoint = torch.load(model_file)
state_dict = checkpoint['model']

new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

with open('data/enc.pkl', 'rb') as f:
    enc_pickled = pickle.load(f)
enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)

lm = NanoLM(
    model=model, 
    config=config, 
    enc=enc, 
)

#############

input_str = "STANTIAL EQUIVALENCE DETERMINATION DECISION SUMMARY A. 510(k) Number: K153137 B. Purpose for Submission: Clearance of a new device C. Measurand: Anti-PF4/Heparin Total Antibodies D. Type of Test: Automated, latex enhanced immuno-turbidimetric assay E. Applicant: Instrumentation Laboratory (IL) Co. F. Proprietary and Established Names: HemosIL HIT‐Ab(PF4‐H) HemosIL HIT‐Ab(PF4‐H) Controls G. Regulatory Information: 1. Regulation section: 21 CFR 864.7695, Platelet factor 4 radioimmunoassay 21 CFR 864.5425, Multipurpose system for in vitro coagulation studies 2. Classification: Class II 3. Product code: 2 LCO, Platelet factor 4 radioimmunoassay GGN, Plasma, Coagulation Control 4. Panel: Hematology (81) H. Intended Use: 1. Intended use(s): HemosIL HIT-Ab(PF4-H) is a qualitative, fully automated, latex enhanced immunoassay for the detection of anti-platelet factor 4/heparin (PF4/H) antibodies. The assay is for use in human 3.2% or 3.8% citrated plasma on the ACL TOP® Family of instruments in a laboratory setting. The result provided by the assay should be interpreted as either positive or negative based on the assay cut-off (1.0 U/mL). The positive or negative result aids in determining the risk for heparin induced thrombocytopenia (HIT) when used in conjunction with other laboratory and clinical findings. Anti-PF4/Heparin antibodies are commonly found in patients with HIT. For use in adult population suspected of HIT. Not for use in isolation to exclude HIT. HemoslL HIT-Ab(PF4-H) Controls are for the Quality Control of the HemosIL HIT-Ab(PF4-\nH) assay as performed on the ACL TOP® Family of instruments. For prescription use. 2. Indication(s) for use: Same as Intended Use 3. Special conditions for use statement(s): For prescription use 4. Special instrument requirements: ACL TOP® Family Instruments I. Device Description: The HemosIL HIT-Ab(PF4-H) kit is a latex particle enhanced immuno-turbidimetric assay to detect total anti‐PF4/Heparin antibodies found in HIT patients. A monoclonal 3 antibody that mimics human HIT antibodies is coated onto latex particles. The HemosIL HIT-Ab(PF4-H) kit consists of: Latex Reagent: Suspension of polystyrene latex particles coated with purified mouse monoclonal anti-PF4-Heparin in Tris buffer, containing bovine serum albumin, stabilizers and preservative. Stabilizer: PBS buffer containing bovine serum albumin, stabilizers and preservative. Complex: Solution of PF4-PVS complex (PF4 from human platelets complexed to PVS), in PBS buffer containing bovine serum albumin, stabilizers and preservative. Contains 0.02% Bronidox™ as a preservative. Calibrator: Lyophilized solution of a monoclonal anti- PF4-Heparin antibody in Tris buffer containing bovine serum albumin, stabilizers and preservative. Controls: The Low and High HIT‐Ab(PF4‐H) Controls are prepared by means of a dedicated process and contain different concentrations of humanized monoclonal anti‐PF4‐Heparin human IgG. • Low HIT Control: Control intended for the assessment of precision and accuracy of the assay at PF4/H antibody levels at or below the cut‐off. • High HIT Control: Control intended for the assessment of precision and accuracy of the assay at abnormal PF4/H antibody levels. J. Substantial Equivalence Information: 1. Predicate device name(s): Asserachrom HPIA Test kit from Diagnostica Stago 2. Predicate 510(k) number(s): K003767 3. Comparison with predicate: 4 Similarities Item Device Predicate Trade Names HemosIL HIT-Ab(PF4-H) HemosIL HIT-Ab(PF4-H) Controls (K153137) Asserachrom HPIA Test Kit (kit includes two control levels) (K003767) Measurand Anti-PF4/Heparin Total Antibodies Anti‐PF4/Heparin Total Antibodies Detection Method Absorbance (Turbimetric) Absorbance (Colorimetric) Intended Use HemosIL HIT-Ab(PF4-H) is a qualitative, fully automated, latex enhanced immunoassay for the detection of anti-platelet factor 4/heparin (PF4/H) antibodies. The assay is for use in human 3.2% or 3.8% citrated plasma on the ACL TOP® Family of instruments in a laboratory setting. The result provided by the assay should be interpreted as either positive or negative based on the assay cut-off (1.0 U/mL). The positive or negative result aids in determining the risk for heparin induced thrombocytopenia (HIT) when used in conjunction with other laboratory and clinical findings. Anti-PF4/Heparin antibodies are commonly found in patients with HIT. For use in adult population suspected of HIT. Not for use in isolation to exclude HIT. HemosIL HIT-Ab(PF4-H) Controls are for the Quality Control of the HemosIL HIT-Ab(PF4-H) assay as performed on the ACL TOP Family of instruments. For prescription use. The ASSERACHROM® HPIA Test Kit is intended for use as a qualitative procedure for the detection of anti‐heparin‐platelet factor 4 (anti-Heparin-PF4) antibodies in citrated plasma or serum by the sandwich technique of enzyme-linked immunosorbent assay (ELISA). The presence in plasma or serum of anti-Heparin-PF4 antibodies, together with a concurrent drop in platelet count, is generally associated with Type II heparin‐induced thrombocytopenia (Type II HIT), a condition that occurs during heparin therapy, leading to arterial or venous thrombosis. Assay Type Qualitative Qualitative Differences Item Device Predicate Sample Types Citrated human plasma only Citrated human plasma or serum Cut‐off Fixed clinical cut‐off: ≥ 1.0 U/mL Variable clinical cut‐off Cut‐off is lot and plate dependent. Every time a plate is processed, the cut‐off for this plate is calculated as the percentage (X%) of the value 5 Differences Item Device Predicate obtained for the reagent supplied with the kit. This percentage is provided for each lot through the insert sheets. Methodology Latex‐enhanced immuno-turbidimetric assay Two‐step enzyme immunoassay (EIA) sandwich method with a final colorimetric detection. Antibodies Purified mouse monoclonal anti-PF4-Heparin Goat anti-human antibodies to IgG, IgA and IgM Controls Controls sold separately: - Low Level at or below the cut-off - High Level at abnormal anti-PF4/H antibody level. Controls included in test kit: - Negative level - Positive level Calibrator Traceability The reported values for the kit calibrator are determined over multiple runs on the ACL TOP Family of instruments using specific lots of reagents and against an internal House Standard. Since an HIT International Standard is not currently available, arbitrary units (U/mL) have been established. Not Applicable K. Standard/Guidance Document Referenced (if applicable): EP05-A3; Evaluation of Precision of Quantitative Measurement Procedures; Approved Guideline; 2014 EP06-A; Evaluation of the Linearity of Quantitative Measurement Procedures; a Statistical Approach; Approved Guideline; 2003 EP07-A2; Interference Testing in Clinical Chemistry; Approved Guideline; 2005 EP09-A3; Measurement Procedure Comparison and Bias Estimation Using Patient Samples; Approved Guideline; 2013 EP12-A2; User Protocol for Evaluation of Qualitative Test Performance; Approved Guideline; 2008 EP14-A3; Evaluation of Commutability of Processed Samples; Approved Guideline; 2013 EP17-A2; Evaluation of Detection Capability For Clinical Laboratory Measurement Procedures; Approved Guideline; 2012 EP24-A2; Assessment of Diagnostic Accuracy of Laboratory Tests Using receiver Operating 6 Characteristic Curves; Approved Guideline; 2011 EP25-A3; Evaluation of Stability of In Vitro Diagnostic Reagents; Approved Guideline; 2009 EP28-A3C; Defining, Establishing and Verifying Reference Intervals in the Clinical Laboratory; Approved Guideline; 2010 L. Test Principle: The HemosIL HIT-Ab(PF4-H) kit is a latex particle enhanced immuno-turbidimetric assay to detect total Anti‐PF4/Heparin (PF4/H) antibodies found in HIT patients. A monoclonal antibody that mimics human HIT antibodies\nMeasurand:"
input_enc = enc.encode(input_str)

print(input_enc)
print(len(input_enc))

stop_tokens = ['\n', '.']
stop_tokens = [enc.encode(token)[0] for token in stop_tokens]
print(stop_tokens)

with ctx:
    output_enc = lm.generate([torch.tensor(input_enc)], n_tokens=[48], samples=[False], temperatures=[1], stop_tokens=[stop_tokens])
    #output_enc = lm.generate_naive(torch.tensor(input_enc).unsqueeze(0), n_tokens=48, sample=True, temperature=1)

#print(output_enc[0])
print(enc.decode(output_enc[0].tolist()))

