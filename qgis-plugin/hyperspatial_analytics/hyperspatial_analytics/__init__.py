# -*- coding: utf-8 -*-

def classFactory(iface):
    from .plugin import HyperspatialAnalyticsPlugin
    return HyperspatialAnalyticsPlugin(iface)
