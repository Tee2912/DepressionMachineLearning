# Generated by Django 2.0.2 on 2018-04-21 23:17

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='mlmodel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('date_created', models.DateTimeField(default=django.utils.timezone.now)),
                ('precision', models.PositiveIntegerField(default=0)),
                ('recall', models.PositiveIntegerField(default=0)),
                ('accuracy', models.PositiveIntegerField(default=0)),
                ('truepositiverate', models.PositiveIntegerField(default=0)),
                ('falsepositiverate', models.PositiveIntegerField(default=0)),
                ('tn', models.PositiveIntegerField(default=0)),
                ('tp', models.PositiveIntegerField(default=0)),
                ('fn', models.PositiveIntegerField(default=0)),
                ('fp', models.PositiveIntegerField(default=0)),
                ('csvfile', models.CharField(max_length=256)),
                ('locked', models.BooleanField(default=False)),
            ],
        ),
        migrations.DeleteModel(
            name='Bucketlist',
        ),
    ]
