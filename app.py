__all__ = ['learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'intf']

import gradio as gr
from fastai.vision.all import *
from PIL import Image

learn = load_learner('model.pkl')

categories = ("Acid", "Base", "Neutral", "Zwitterion")

def classify_image(SMILES, img):
    if SMILES != '':
        mol = Chem.MolFromSmiles(SMILES)
        Chem.Draw.MolToFile(mol, f'./new_images/{SMILES}.png') 
        img = PILImage.create(f'./new_images/{SMILES}.png')
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs))), img


image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
path = './test/'
examples = [
    ["O=C(C)Oc1ccccc1C(=O)O", f"{path}0.png"],
    ["C1CN2CCN1CC2 N12CCN(CC1)CC2", f"{path}1.png"],
    ["CC(O)=O", f"{path}2.png"],
    ["CCN(CC)CC", f"{path}3.png"],
    ["C1=CC2=C(C=C1O)C(=CN2)CCN", f"{path}4.png"],
    ["OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O", f"{path}5.png"],
    ["OC(=O)CC1CNCC2=C1C=CC=C2", f"{path}6.png"],
    ["CN1CC(CN2C3=CC=C(Cl)C=C3C=NCC2=O)C1", f"{path}7.png"],
    ["OC(=O)C1CN2CCC1CC2", f"{path}8.png"],
    ["CS(=O)(=S)NC(=O)C1CC2CCC1CC2", f"{path}9.png"],
]

intf = gr.Interface(
    fn=classify_image, 
    inputs=[gr.Textbox(lines=1, placeholder="Enter SMILES String Here..."), image], 
    outputs=[label, image],
    examples=examples)

intf.launch(inline=False)
