import os

content = ['brad_pitt', 'golden_gate', 'hoovertowernight', 'IMG_4343', 'tubingen']
style = [ 'frida_kahlo', 'picasso_selfport1907', 'seated-nude', 'shipwreck', 'starry_night', 'the_scream', 'woman-with-hat-matisse']
style_weights = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
kernels = ['linear', 'poly', 'gaussian']


def gen_html(out_name):
    fout = open(out_name, "w")
    text = ''
    for c in content:
        for s in style:
            for k in kernels:
                line = '<tr><td>%s</td><td><img src="%s.jpg" alt="global-1" style="width: 200"></td>\
                        <td><img src="%s.jpg" alt="global-1" style="width: 200"></td>' % (k, c, s)
                for w in style_weights:
                    image = '%s-%s-%s-%.2f-1.00.jpg' % (c, s, k, w)
                    line += '<td><img src="%s" alt="global-1" style="width: 200"><p align=center>%.2f</p></td>' % (image, w)
                line += '</tr>\n'
                text += line 
                print line
    html_text = '<html><head><meta charset="UTF-8"></head><body bgcolor="#aaaaaa">\
        <div align="center"><table>%s</div>\
        </table></body></html>' % text

    print html_text

    fout.write(html_text)

    fout.close()


if __name__ == '__main__':
    gen_html('output/show.html')

