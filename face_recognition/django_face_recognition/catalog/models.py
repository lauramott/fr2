# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models


# Create your models here.
class HomePage(models.Model):
    user_name = models.TextField(max_length=50)


from django.core.files.base import ContentFile


class PhotoModel(models.Model):
    video = models.FileField(upload_to='catalog/static/images')

    def save_file(request):
        photomodel = PhotoModel.objects.get(id=1)
        file_content = ContentFile(request.FILES['video'].read())
        photomodel.video.save(request.FILES['video'].name, file_content)
