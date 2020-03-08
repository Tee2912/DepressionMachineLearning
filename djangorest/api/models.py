from django.db import models
from django.utils import timezone

# Create your models here.
class mlmodel(models.Model):
    name = models.CharField(max_length=255, blank=False, unique=True)
    date_created = models.DateTimeField(default=timezone.now)
    precision = models.FloatField(default=0)
    recall = models.FloatField(default=0)
    f1score = models.FloatField(default=0)
    accuracy = models.FloatField(default=0)
    truepositiverate = models.FloatField(default=0)
    falsepositiverate = models.FloatField(default=0)
    tn = models.PositiveIntegerField(default=0)
    tp = models.PositiveIntegerField(default=0)
    fn = models.PositiveIntegerField(default=0)
    fp = models.PositiveIntegerField(default=0)
    csvfile = models.CharField(max_length=256)
    locked = models.BooleanField(default=False)



    def __str__(self):
        return self.name
