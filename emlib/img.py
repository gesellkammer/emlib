from PIL import Image
import os.path
from io import BytesIO

from typing import Union as U, Tuple


def imgSize(path:str) -> Tuple[int, int]:
    """ returns (width, height) """
    im = Image.open(path)
    return im.size


def asImage(obj: U[str, Image.Image]) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj
    elif isinstance(obj, str):
        return Image.open(obj)
    else:
        raise TypeError(f"obj type {type(obj)} not supported")


def hasTransparency(img: U[str, Image.Image]) -> bool:
    img = asImage(img)
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True
    return False


def removeTransparency(im: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """
    Remove transparency (alpha channel) from a PIL image

    Args:
        im: a PIL image (read via PIL.Image.open)
        background: the color to use as background

    Returns:

    """
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        # Need to convert to RGBA if LA format due to a bug in
        # PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').getchannel('A')  # .split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, background + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


def pngRemoveTransparency(pngfile: str, outfile: str = None, background=(255, 255, 255)
                          ) -> None:
    '''
    Remove transparency from a png file

    If outfile is not given, the operation is performed in place

    Args:
        pngfile: the source file
        outfile: if given, the result is saved to this file
        background: the color to use as replacement for the background

    '''
    assert os.path.splitext(pngfile)[1] == '.png'
    img = Image.open(pngfile)
    img = removeTransparency(img, background=background)
    if outfile:
        img.save(outfile)
    else:
        # instead of overwriting, we send the old file to trash
        import send2trash
        send2trash.send2trash(pngfile)
        img.save(pngfile)


def readImageAsBase64(imgpath:str, outformat:str=None, removeAlpha=False
                      ) -> Tuple[bytes, int, int]:
    """
    Read an image and output its base64 representation

    Args:
        imgpath: the path to the image
        outformat: the format to save the image to

    Returns:
        a tuple (data, width, height), where data is the base64
        representation of the image, as bytes
    """
    import base64
    if not outformat:
        outformat = os.path.splitext(imgpath)[1][1:].lower()
    assert outformat in {'jpeg', 'png'}
    buffer = BytesIO()
    im = Image.open(imgpath)
    if removeAlpha:
        im = removeTransparency(im)
    im.save(buffer, format=outformat)
    imgbytes = base64.b64encode(buffer.getvalue())
    width, height = im.size
    return imgbytes, width, height


def htmlImgBase64(imgpath: str, width:U[int, str]=None, maxwidth:U[int,str]=None,
                  margintop='14px', padding='10px',
                  removeAlpha=False, scale:float=1.
                  ) -> str:
    """
    Read an image and return the data as base64 within an img html tag
    
    Args:
        imgpath: the path to the image 
        width: the width of the displayed image. Either a width
            in pixels or a str as passed to css ('800px', '100%').
        maxwidth: similar to width
        scale: if width is not given, a scale value can be used to display
            the image at a relative width

    Returns:
        the generated html
    """
    imgstr, imwidth, imheight = readImageAsBase64(imgpath, outformat='png',
                                                  removeAlpha=removeAlpha)
    imgstr = imgstr.decode('utf-8')
    if scale and not width:
        width = imwidth * scale
    attrs = [f'padding:{padding}',
             f'margin-top:{margintop}']
    if maxwidth:
        if isinstance(maxwidth, int):
            maxwidth = f'{maxwidth}px'
        attrs.append(f'max-width: {maxwidth}')
    if width is not None:
        if isinstance(width, int):
            width = f'{width}px'
        attrs.append(f'width:{width}')
    style = ";\n".join(attrs)
    return fr'''
        <img style="display:inline; {style}"
             src="data:image/png;base64,{imgstr}"/>'''