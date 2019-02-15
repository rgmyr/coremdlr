"""
Configuration of lithological facies classes, their attributes, and visualization properties.
"""
from striplog import Decor, Component, Legend

###++++++++++++++++++++###
### Facies Definitions ###
###++++++++++++++++++++###

facies = {
    'nc': Component({'lithology': 'none', 'train':'n'}),
    'bs': Component({'lithology': 'bad-sandstone', 'train': 'n'}),
    's': Component({'lithology': 'sandstone', 'train':'y'}),
    'is': Component({'lithology': 'clay-prone sandstone', 'train':'y'}),
    'ih': Component({'lithology': 'sandy mudstone', 'train':'y'}),
    'os': Component({'lithology': 'oilstained', 'train':'y'}),
    'sh': Component({'lithology': 'mudstone', 'train':'y'}),
    #'t': Component({'lithology': 'turbidite', 'train':'y'}),
}

lithologies = [c.lithology for c in facies.values()]
lithologies_dict = {k: v.lithology for k, v in facies.items()}

def lithology_to_key(lithology_str):
    """Take str representation of lithology component, return the corresponding `facies` key."""
    try:
        litho = lithology_str.split(',')[0].lower()
        idx = lithologies.index(litho)
        return list(facies.keys())[idx]
    except IndexError:
        raise ValueError('{} is not present in lithologies: {}'.format(lithology_str, lithologies))


collapsed_facies = [Component({'lithology': 'sandstone', 'train':'y'}),
                    Component({'lithology': 'clay-prone sandstone', 'train':'y'}),
                    Component({'lithology': 'sandy mudstone', 'train':'y'}),
		    Component({'lithology': 'oilstained', 'train':'y'}),
                    Component({'lithology': 'mudstone', 'train':'y'})]


###########################
### Visualization Style ###
###########################

nocore = Decor({
    'component': facies['nc'],
    'colour': 'white',
    'hatch': '/',
    'width': '5',
})

badsand = Decor({
    'component': facies['bs'],
    'colour': 'orange',
    'hatch': '.',
    'width': '4',
})

sandstone = Decor({
    'component': facies['s'],
    'colour': 'yellow',
    'hatch': '.',
    'width': '4',
})

clay_prone_sandstone = Decor({
    'component': facies['is'],
    'colour': 'greenyellow',
    'hatch': '--',
    'width': '3',
})

sandy_mudstone = Decor({
    'component': facies['ih'],
    'colour': 'darkseagreen',
    'hatch': '---',
    'width': '2',
})

oilstained = Decor({
    'component': facies['os'],
    'colour': 'brown',
    'hatch': '/',
    'width': '3',
})

mudstone = Decor({
    'component': facies['sh'],
    'colour': 'darkgray',
    'hatch': '-',
    'width': '1',
})

legend = Legend([nocore, badsand, sandstone, clay_prone_sandstone, sandy_mudstone, oilstained, mudstone])#, turbidite])
collapsed_legend = Legend([sandstone, clay_prone_sandstone, sandy_mudstone, oilstained, mudstone])

# Not sure about the best way to do this, probably better
# just to omit those intervals completely.
#turbidite = Decor({
#    'component': facies['t'],
#    'colour': 'green',
#    'hatch': 'xxx',
#    'width': '3',
#})
