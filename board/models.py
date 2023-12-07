from django.db import models


# Create your models here.
class Board(models.Model):
    title = models.CharField(max_length=30)
    content = models.TextField()
    created_at = models.DateField

    class Meta:
        ordering = ["-pk"]
