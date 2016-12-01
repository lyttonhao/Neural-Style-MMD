content = ['brad_pitt', 'golden_gate', 'hoovertowernight', 'IMG_4343', 'tubingen']
style = [ 'frida_kahlo', 'picasso_selfport1907', 'seated-nude', 'shipwreck', 'starry_night', 'the_scream', 'woman-with-hat-matisse']
style_weights = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
kernels = ['linear', 'poly', 'gaussian']
gpus = [5, 6, 7]


fo = [open('run-all-%d.sh' % g, 'w') for g in gpus]
i = 0
for c in content:
    for s in style:
        for k in kernels:
            for w in style_weights:
                i = (i + 1) % 3
                fo[i].write('python run.py --mmd-kernel %s --gpu %d --style-weight %f --content-image input/%s.jpg --style-image input/%s.jpg --output %s-%s\n' % (k, gpus[i], w, c, s, c, s))
                
            
    
