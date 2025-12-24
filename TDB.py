
!pip install -q transformers datasets gradio PyPDF2 catboost shap xgboost

import random
import os
import zipfile
import re
from collections import defaultdict

import torch
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

import PyPDF2
import gradio as gr
from google.colab import files

labels = ["FCC", "BCC", "LIQUID", "SOLID", "SUBLATTICE", "NONE"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

phase_sentences = {
"FCC": [
# ===== original 20 =====
"The alloy adopts an FCC structure.",
"FCC phase was observed in the micrograph.",
"An FCC matrix dominates the structure.",
"XRD indicates a face-centered cubic arrangement.",
"The compound crystallizes in an FCC lattice.",
"The FCC phase remains stable at intermediate temperatures.",
"Grains with FCC symmetry were identified using EBSD.",
"FCC ordering contributes to enhanced ductility.",
"The dominant crystallographic phase is FCC.",
"A single-phase FCC solid solution was obtained.",
"FCC reflections appear prominently in the diffraction pattern.",
"The alloy exhibits FCC symmetry after annealing.",
"Transmission electron microscopy confirms an FCC lattice.",
"The FCC phase persists after rapid quenching.",
"A homogeneous FCC phase was achieved.",
"FCC-based solid solutions are favored at high entropy.",
"The FCC structure enhances plastic deformation.",
"Thermodynamic calculations predict FCC stability.",
"FCC grains grow during solution treatment.",
"The material retains an FCC structure upon cooling.",

# ===== new 40 =====
"The FCC lattice provides high atomic packing efficiency.",
"FCC symmetry was confirmed through Rietveld refinement.",
"The material shows FCC ordering across all compositions.",
"FCC crystal structure dominates after homogenization.",
"An FCC solid solution forms during slow cooling.",
"The FCC phase improves formability of the alloy.",
"FCC stability is enhanced by alloying additions.",
"The FCC phase exhibits low lattice distortion.",
"FCC ordering was maintained under mechanical loading.",
"The FCC structure allows multiple slip systems.",
"Electron backscatter patterns match FCC symmetry.",
"The FCC phase persists across a wide temperature range.",
"FCC grains display equiaxed morphology.",
"The FCC lattice minimizes internal strain energy.",
"FCC structure is favored due to entropy stabilization.",
"The FCC phase dominates the equilibrium microstructure.",
"FCC ordering suppresses brittle fracture.",
"The FCC lattice shows uniform atomic distribution.",
"FCC symmetry was verified using diffraction analysis.",
"The FCC phase supports high ductility at room temperature.",
"FCC phase formation was predicted by CALPHAD modeling.",
"The FCC lattice remains intact after deformation.",
"FCC grains exhibit low anisotropy.",
"FCC phase contributes to enhanced toughness.",
"The FCC structure stabilizes at lower cooling rates.",
"FCC ordering resists phase transformation.",
"FCC lattice parameters vary with composition.",
"The FCC phase controls the mechanical response.",
"FCC crystallography governs slip behavior.",
"FCC domains coalesce during annealing.",
"The FCC lattice dominates the final microstructure.",
"FCC phase suppresses martensitic transformation.",
"FCC structure enhances dislocation mobility.",
"FCC grains are uniformly distributed.",
"FCC phase remains stable under cyclic loading.",
"FCC ordering was confirmed experimentally.",
"The FCC lattice persists after thermal cycling.",
"FCC symmetry dictates deformation mechanisms.",
"The FCC phase defines the alloy matrix."
],

"BCC": [
# ===== original 20 =====
"BCC phase appears above 1200K.",
"Body-centered cubic phase dominates in the sample.",
"The BCC phase forms after heat treatment.",
"XRD pattern confirms BCC symmetry.",
"The microstructure shows BCC grains.",
"BCC structure becomes stable at elevated temperatures.",
"The alloy transforms from FCC to BCC upon heating.",
"Neutron diffraction confirms a BCC lattice.",
"The presence of BCC phase increases strength.",
"BCC domains are observed in the microstructure.",
"A single-phase BCC solid solution was detected.",
"BCC ordering is favored at lower atomic packing.",
"The material exhibits BCC symmetry after quenching.",
"Phase-field simulations predict BCC formation.",
"BCC grains coarsen during prolonged annealing.",
"The BCC phase contributes to higher yield strength.",
"Electron diffraction patterns indicate BCC ordering.",
"BCC lattice parameters were extracted from XRD.",
"The alloy stabilizes in a BCC phase at high temperature.",
"BCC regions coexist with FCC in a dual-phase structure.",

# ===== new 40 =====
"BCC lattice becomes dominant under thermal activation.",
"The BCC phase exhibits reduced ductility.",
"BCC symmetry governs high-temperature deformation.",
"The BCC phase nucleates during phase transformation.",
"BCC ordering strengthens the alloy matrix.",
"BCC grains elongate during deformation.",
"The BCC structure is stable at elevated entropy.",
"BCC lattice shows lower atomic packing factor.",
"BCC phase formation increases hardness.",
"The BCC structure controls creep resistance.",
"BCC grains align along deformation direction.",
"BCC ordering suppresses slip activity.",
"The BCC lattice favors diffusion-controlled processes.",
"BCC symmetry emerges after rapid heating.",
"The BCC phase dominates at high temperature regimes.",
"BCC ordering enhances yield strength.",
"The BCC lattice distorts under applied stress.",
"BCC domains expand during annealing.",
"The BCC phase reduces fracture toughness.",
"BCC grains display columnar morphology.",
"BCC symmetry influences dislocation motion.",
"The BCC structure stabilizes under pressure.",
"BCC phase formation was observed experimentally.",
"The BCC lattice responds strongly to alloying.",
"BCC ordering increases lattice friction.",
"The BCC phase governs high-temperature strength.",
"BCC lattice parameters change with temperature.",
"BCC symmetry limits available slip systems.",
"The BCC phase promotes strengthening mechanisms.",
"BCC grains grow during thermal exposure.",
"The BCC lattice dominates after transformation.",
"BCC ordering reduces atomic mobility.",
"BCC phase stabilizes at high temperatures.",
"The BCC structure affects diffusion rates.",
"BCC symmetry persists after cooling.",
"BCC grains exhibit anisotropic behavior.",
"BCC phase influences mechanical anisotropy.",
"BCC ordering controls phase stability.",
"The BCC lattice defines microstructural evolution."
],

"LIQUID": [
# ===== original 20 =====
"A liquid phase was observed during melting.",
"Upon heating, the material enters a liquid phase.",
"The alloy remains liquid above 1500°C.",
"Molten state was confirmed using DSC.",
"Solid to liquid transition occurs near 1800K.",
"The material exists as a homogeneous liquid at high temperature.",
"A fully liquid phase forms during alloy processing.",
"The system transitions into a liquid region upon heating.",
"Liquid phase stability was confirmed thermodynamically.",
"The sample becomes fully molten beyond the liquidus temperature.",
"The alloy exhibits complete melting at elevated temperatures.",
"A stable liquid phase persists over a wide temperature range.",
"Liquid behavior dominates above the melting point.",
"The presence of a liquid phase was detected experimentally.",
"High-temperature treatment results in a liquid state.",
"The liquid phase shows uniform elemental distribution.",
"Molten alloy properties were measured in situ.",
"The system remains in a liquid phase during heating.",
"A single liquid phase was observed under equilibrium conditions.",
"Liquid phase formation precedes solidification.",

# ===== new 40 =====
"The alloy exists entirely in a molten state.",
"The liquid phase exhibits homogeneous mixing.",
"Liquid stability extends across a broad composition range.",
"The molten phase dominates the phase diagram.",
"Liquid formation occurs beyond the solidus line.",
"The liquid phase shows high atomic mobility.",
"Complete liquefaction was observed experimentally.",
"The liquid state facilitates elemental diffusion.",
"Liquid phase behavior governs casting processes.",
"The molten alloy exhibits uniform density.",
"Liquid phase evolution was tracked during heating.",
"The system remains molten under sustained heating.",
"Liquid phase formation is reversible upon cooling.",
"The liquid region expands with increasing temperature.",
"Liquid stability was confirmed through simulations.",
"Molten alloy exhibits low viscosity at high temperature.",
"The liquid phase promotes chemical homogeneity.",
"Liquid behavior dominates thermal response.",
"The molten state persists until solidification begins.",
"Liquid phase exists above the critical temperature.",
"The alloy enters a fully liquid regime.",
"The liquid state allows rapid atomic rearrangement.",
"Liquid phase formation reduces mechanical rigidity.",
"The molten alloy flows under gravity.",
"Liquid stability is influenced by composition.",
"The liquid phase precedes nucleation.",
"Liquid behavior controls solidification kinetics.",
"Liquid phase was sustained during thermal cycling.",
"The molten alloy shows isotropic behavior.",
"Liquid phase dominates processing conditions.",
"The alloy transitions fully into a liquid state.",
"Liquid formation was observed macroscopically.",
"The molten phase exhibits high entropy.",
"Liquid phase governs melt-solid interactions.",
"The system achieves a stable liquid configuration.",
"Liquid behavior was captured in thermal analysis.",
"Molten state enables casting and shaping.",
"The liquid phase defines high-temperature behavior.",
"Liquid stability determines melting characteristics."
],

"SOLID": [
# ===== original 20 =====
"The system stabilizes in a solid phase.",
"The alloy exists as a solid at room temperature.",
"The sample remains solid during cooling.",
"A stable solid phase was detected.",
"The structure is predominantly solid below 1000K.",
"The material retains a solid state under ambient conditions.",
"A fully solid phase forms after solidification.",
"The alloy remains solid throughout the experiment.",
"Solid phase stability was confirmed by XRD.",
"The system transforms into a solid upon cooling.",
"A single solid phase dominates the microstructure.",
"The solid state persists at low temperatures.",
"Solid phase formation occurs after nucleation.",
"The alloy solidifies into a stable phase.",
"The sample exists entirely in the solid state.",
"Thermodynamic analysis predicts solid phase stability.",
"The solid phase exhibits well-defined grains.",
"Solidification leads to a uniform solid structure.",
"The material remains solid below the transition temperature.",
"Solid-state behavior governs the mechanical response.",

# ===== new 40 =====
"The solid phase dominates at low temperatures.",
"Solidification completes during cooling.",
"The alloy maintains a rigid solid structure.",
"Solid state properties govern deformation.",
"The solid phase exhibits crystalline ordering.",
"Solid phase formation is energetically favored.",
"The solid structure remains stable over time.",
"The alloy transitions fully into a solid.",
"Solid state stability was verified experimentally.",
"The solid phase controls elastic behavior.",
"Solidification results in grain formation.",
"The solid structure persists after processing.",
"Solid phase stability depends on composition.",
"The solid state defines mechanical strength.",
"Solid behavior dominates below melting point.",
"The solid phase restricts atomic mobility.",
"The alloy forms a coherent solid matrix.",
"Solidification occurs after nucleation and growth.",
"The solid structure resists deformation.",
"Solid state governs fracture behavior.",
"Solid phase evolution was monitored during cooling.",
"The alloy exhibits a stable solid framework.",
"Solidification produces a continuous phase.",
"The solid phase maintains lattice integrity.",
"Solid state characteristics influence hardness.",
"The solid structure remains intact under load.",
"Solid phase dominates the microstructure.",
"Solidification leads to phase stabilization.",
"The solid phase governs thermal conductivity.",
"Solid state properties define stiffness.",
"The alloy remains fully solid after cooling.",
"Solid structure shows minimal porosity.",
"The solid phase supports mechanical integrity.",
"Solidification establishes final structure.",
"The solid state resists thermal agitation.",
"Solid phase formation completes at low temperature.",
"The solid structure defines macroscopic properties.",
"Solid state governs elastic-plastic transition.",
"The solid phase remains stable under service conditions."
],

"SUBLATTICE": [
# ===== original 20 =====
"The sublattice model (A,B)(C) was used.",
"Two-sublattice model describes the disordered phase.",
"(Ni)(Al) sublattice was assumed in the simulation.",
"The sublattice configuration used was (Fe,Cr)(C).",
"A ternary sublattice (A,B)(C,D) fits the data.",
"A multi-sublattice approach was adopted for modeling.",
"The phase was described using a compound energy formalism.",
"Sublattice occupancy was optimized during calculation.",
"A two-sublattice structure captures site preferences.",
"The model assumes separate metal and non-metal sublattices.",
"Sublattice interactions influence phase stability.",
"The alloy was modeled using a three-sublattice scheme.",
"Cation and anion sublattices were treated independently.",
"Sublattice disorder was included in the thermodynamic model.",
"A hierarchical sublattice description was employed.",
"Sublattice site fractions were calculated explicitly.",
"The phase stability depends on sublattice occupancy.",
"Multiple sublattices account for chemical ordering.",
"Sublattice formalism enables accurate phase prediction.",
"The crystal structure was represented by coupled sublattices.",

# ===== new 40 =====
"Sublattice models capture atomic site preferences.",
"Sublattice interactions determine thermodynamic behavior.",
"The phase was modeled using coupled sublattices.",
"Sublattice occupancy influences phase stability.",
"A flexible sublattice framework was applied.",
"Sublattice ordering was included in calculations.",
"The model resolves site-specific interactions.",
"Sublattice formalism describes chemical ordering.",
"Sublattice parameters were optimized numerically.",
"Multi-sublattice descriptions improve accuracy.",
"Sublattice disorder affects free energy.",
"Sublattice interactions control phase equilibria.",
"The crystal structure was decomposed into sublattices.",
"Sublattice-based modeling captures ordering effects.",
"Sublattice formalism supports multicomponent systems.",
"Atomic positions were assigned to sublattices.",
"Sublattice interactions define phase boundaries.",
"Sublattice description enables site fraction control.",
"Sublattice models predict ordering transitions.",
"Sublattice configuration impacts stability.",
"Compound energy formalism relies on sublattices.",
"Sublattice interactions affect phase diagrams.",
"Sublattice modeling improves thermodynamic predictions.",
"Sublattice occupancy evolves with temperature.",
"Sublattice disorder influences entropy.",
"Sublattice scheme captures configurational effects.",
"Sublattice formalism allows complex ordering.",
"Sublattice interactions govern phase formation.",
"Sublattice models support CALPHAD calculations.",
"Sublattice description accounts for site competition.",
"Sublattice framework resolves multi-site occupancy.",
"Sublattice modeling enables phase prediction.",
"Sublattice approach captures chemical complexity.",
"Sublattice configuration was refined computationally.",
"Sublattice interactions stabilize ordered phases.",
"Sublattice formalism governs phase energetics.",
"Sublattice model represents crystal chemistry.",
"Sublattice approach improves modeling fidelity."
],

"NONE": [
# ===== original 20 =====
"Samples were polished before testing.",
"Experimental setup included a furnace and crucible.",
"The cooling rate was maintained at 10K/min.",
"Scanning electron microscopy was used.",
"Mechanical testing showed high hardness.",
"The experiment was conducted under argon atmosphere.",
"Specimens were prepared using standard metallographic methods.",
"Chemical composition was measured using EDS.",
"Heat treatment was performed in a vacuum furnace.",
"The sample dimensions were carefully controlled.",
"Temperature was monitored using a thermocouple.",
"Data acquisition was performed automatically.",
"The testing protocol followed standard procedures.",
"Samples were cleaned prior to analysis.",
"Repeated measurements ensured reproducibility.",
"Optical microscopy was used for initial inspection.",
"Sample preparation followed ASTM standards.",
"All experiments were carried out at ambient pressure.",
"The methodology is summarized in the experimental section.",
"Calibration was performed before measurements.",

# ===== new 40 =====
"The experimental procedure followed established protocols.",
"Samples were handled using standard laboratory practices.",
"The furnace temperature was controlled precisely.",
"Measurements were conducted under controlled conditions.",
"The apparatus was calibrated prior to testing.",
"Samples were stored in a dry environment.",
"Experimental uncertainty was minimized.",
"Data processing was performed using standard software.",
"The setup ensured consistent heating conditions.",
"Specimen geometry was standardized.",
"Testing conditions were kept constant.",
"Experimental parameters were carefully selected.",
"Sample labeling was done systematically.",
"Results were recorded digitally.",
"Multiple trials were conducted for accuracy.",
"The experimental workflow was documented.",
"Equipment maintenance was performed regularly.",
"Ambient conditions were monitored.",
"The procedure adhered to safety guidelines.",
"Instrumentation accuracy was verified.",
"Samples were prepared in batches.",
"Testing followed laboratory protocols.",
"Data integrity was ensured throughout.",
"Experimental errors were minimized.",
"The methodology ensured repeatability.",
"Measurements were cross-checked.",
"Sample handling followed best practices.",
"Environmental factors were controlled.",
"The experiment was performed systematically.",
"Data was archived for reference.",
"Testing parameters were logged.",
"Quality control checks were applied.",
"Experimental consistency was maintained.",
"Procedural steps were standardized.",
"Measurements were validated independently.",
"The workflow followed laboratory standards.",
"Results were reproducible.",
"Experimental documentation was complete.",
"Testing was conducted under supervision."
]
}

examples = []
for label, sentences in phase_sentences.items():
    for _ in range(100):
        sentence = random.choice(sentences)
        examples.append({
            "text": sentence,
            "label": label2id[label]
        })

random.shuffle(examples)

dataset = Dataset.from_dict({
    "text": [e["text"] for e in examples],
    "label": [e["label"] for e in examples]
})

dataset = dataset.train_test_split(test_size=0.1)


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)


training_args = TrainingArguments(
    output_dir="./bert_phase_classifier",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

trainer.save_model("./bert_phase_classifier")
tokenizer.save_pretrained("./bert_phase_classifier")

uploaded = files.upload()
zip_path = next(iter(uploaded))
extract_dir = "unary_extracted"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

transition_metals = {
    "Sc": "Scandium", "Ti": "Titanium", "Cr": "Chromium",
    "Mn": "Manganese", "Co": "Cobalt", "Ni": "Nickel",
    "Cu": "Copper", "Zn": "Zinc", "Zr": "Zirconium",
    "Nb": "Niobium", "Mo": "Molybdenum", "Tc": "Technetium",
    "Ru": "Ruthenium", "Rh": "Rhodium", "Pd": "Palladium",
    "Ag": "Silver", "Cd": "Cadmium", "Hf": "Hafnium",
    "Ta": "Tantalum", "Re": "Rhenium", "Os": "Osmium",
    "Ir": "Iridium", "Pt": "Platinum", "Au": "Gold",
    "Hg": "Mercury"
}

allowed_elements = set(transition_metals.keys())
name_to_symbol = {v.lower(): k for k, v in transition_metals.items()}

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def count_elements_and_get_top_two_symbols(pdf_file):
    text = extract_text_from_pdf(pdf_file).lower()
    words = re.findall(r"\b\w+\b", text)

    count_map = defaultdict(int)
    for word in words:
        if word.capitalize() in transition_metals:
            count_map[word.capitalize()] += 1
        elif word in name_to_symbol:
            count_map[name_to_symbol[word]] += 1

    if len(count_map) < 2:
        return None

    return [k for k, _ in sorted(count_map.items(), key=lambda x: x[1], reverse=True)[:2]]


def get_unary_data(element):
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.lower() == f"{element.lower()}.tdb":
                with open(os.path.join(root, file), "r") as f:
                    return f.read()
    raise FileNotFoundError(f"No unary function found for {element}")

def process_pdf_and_return_unary_text(pdf_file):
    try:
        symbols = count_elements_and_get_top_two_symbols(pdf_file)
        if not symbols:
            return "Could not identify two valid transition metals."

        x, y = symbols
        data_x = get_unary_data(x)
        data_y = get_unary_data(y)

        return f"{x} ---\n{data_x}\n\n{y} ---\n{data_y}"

    except Exception as e:
        return str(e)


gr.Interface(
    fn=process_pdf_and_return_unary_text,
    inputs=gr.File(label="Upload PDF"),
    outputs=gr.Textbox(label="Top 2 Unary Functions"),
    title="Unary Function Extractor (PDF)",
    description="Upload a PDF. The app detects the two most mentioned transition metals and returns their unary TDB functions."
).launch()
