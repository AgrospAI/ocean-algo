def format_filename(filename, length=40):
    if len(filename) > length:
        return filename[:length - 3] + '...'
    else:
        return filename.ljust(length)
