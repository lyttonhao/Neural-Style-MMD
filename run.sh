#linear
python neural-style.py --mmd-kernel linear --gpu 0 --style-weight 5 --content-image input/brad_pitt.jpg --style-image input/starry_night.jpg --output brad_pitt-starry_night --output-folder output_images
#poly
python neural-style.py --mmd-kernel poly --gpu 0 --style-weight 5 --content-image input/brad_pitt.jpg --style-image input/starry_night.jpg --output brad_pitt-starry_night --output-folder output_images
#gaussian
python neural-style.py --mmd-kernel gaussian --gpu 0 --style-weight 5 --content-image input/brad_pitt.jpg --style-image input/starry_night.jpg --output brad_pitt-starry_night --output-folder output_images
#bn
python neural-style.py --bn-loss --gpu 0 --style-weight 1.000000 --content-image input/brad_pitt.jpg --style-image input/starry_night.jpg --output brad_pitt-starry_night --output-folder output_images

#bn + poly
python neural-style.py --bn-loss --mmd-kernel poly --gpu 0 --style-weight 5 --content-image input/brad_pitt.jpg --style-image input/candy.jpg --output brad_pitt-candy --output-folder output_images --multi-weight 0.5,0.5
#linear + Gaussian
python neural-style.py --mmd-kernel linear,gaussian --gpu 0 --style-weight 5 --content-image input/brad_pitt.jpg --style-image input/candy.jpg --output brad_pitt-candy --output-folder output_images --multi-weight 0.5,0.5
