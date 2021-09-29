import argparse
from matplotlib import pyplot
import os
from tqdm import tqdm

from extraction_utils import read_words

parser = argparse.ArgumentParser()

parser.add_argument('--experiment_id', required=True, \
                    choices=['one', 'two'], \
                    help='Which experiment?')
args = parser.parse_args()

it_words, en_words = read_words(args)

print('Now collecting images arrays...')

images_path = os.path.join('..', '..', 'resources', 'stimuli_images') 
os.makedirs(images_path, exist_ok=True)

for w in tqdm(it_words):
    # Build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax = pyplot.gca()
    p = pyplot.Rectangle((left, bottom), width, height, alpha=0., fill=False)
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)
    ax.add_patch(p)

    ax.text(0.5 * (left + right), 0.5 * (bottom + top), w,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, fontsize='xx-large')

    pyplot.axis('off')
    pyplot.savefig(os.path.join(images_path, '{}.png'.format(w)))
    pyplot.clf()
