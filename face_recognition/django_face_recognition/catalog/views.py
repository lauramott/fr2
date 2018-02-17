# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.views import generic
from django.shortcuts import render
from .models import HomePage, PhotoModel


class HomePage(generic.ListView):
    model = HomePage


class PhotoModel(generic.ListView):
    model = PhotoModel

