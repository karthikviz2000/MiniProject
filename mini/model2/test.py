import json

rawMaterials={'gulab':
{'Khoya gms':100,
'Refined flour tbsp':1,
'Baking soda tbsp':0.25,
'Sugar cups':2,
'Milk tbsp':2,
'cardamoms':4
},'friedRice':
{'boiled rice bowl':1,
'oil tbsp':1,
'garlic cloves':3,
'red chilli':1,
'carrot tbsp':1,
'baby corn':4,
'diced cabbage tbsp':5,
'chopped beans':5,
},'whiteRice':{'rice cup':1},
'biriyani':
{'basmati rice cup':1,
'sliced onion':1,
'green chilli chopped':1,
'ginger garlic paste tbsp':1,
'chopped pudina tbsp':2,
'chopped coriander tbsp':2,
'red chilli powder tbsp':0.75,
'garam masala tbsp':1,
'tomato':1,
'ghee tbsp':2,
'mixed veggies cup':1
},'chapathi':
{
'wheat flour cups':0.5,
'ghee tbsp':0.25
},'vegCurry':
{
'tablespoon vegetable oil tbsp':0.5,
'bay leaf':0.25,
'tablespoon ginger tbsp':0.75,
'cumin tbsp':0.5,
'chopped tomato tbsp':3,
'potato pounds':0.125,
'chopped cauliflower cups':0.5,
'brinjal cups':0.5,
'beans tbsp':3,
'sugar tbsp':0.5,
'turmeric tbsp':0.25,
'sweet potato cups':0.25,
'red chilli tbsp':0.75,
'garam masala tbsp':0.25
},'channaMasala':
{
'chickpeas cups':0.5,
'green chilli':1,
'cloves garlic':1,
'coriander powder tbsp':0.75,
'red chilli tbsp':0.25,
'sunflower oil tbsp':1,
'onion':0.5,
'tomato cups':0.75,
'ginger inches':0.25,
'turmeric tbsp':0.25,
'cumin powder tbsp':0.75,
'garam masala tbsp':0.25,
},'pickle':
{
'cauliflower tbsp':3,
'carrot tbsp':3,
'red chilli tbps':0.5,
'turmeric tbsp':0.25,
'peas tbsp':3,
'mustard seeds tbsp':0.5,
},'chips':
{
'potato':2,
'olive oil cups':0.25
},'pakoda':
{
'onion':0.75,
'cloves garlic':0.5,
'mustard seeds tbsp':0.5,
'spinach bunches':0.5,
'ghee tbsp':2,
'fenugreek seeds tbsp':0.5,
'green chilli':0.5
},'iceCream':{'scoops':1},'savouries':
{
'mixture cups':1
}}

with open('rawMaterials.json', 'w') as outfile:
    json.dump(rawMaterials, outfile,indent=4)