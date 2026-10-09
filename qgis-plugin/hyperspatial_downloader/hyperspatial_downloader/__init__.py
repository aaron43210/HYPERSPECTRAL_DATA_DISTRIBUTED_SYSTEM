# -*- coding: utf-8 -*-

def classFactory(iface):
    from .plugin import HyperspatialDownloaderPlugin
    return HyperspatialDownloaderPlugin(iface)
