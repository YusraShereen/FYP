# Generated by Django 3.2 on 2021-06-27 04:56

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0024_auto_20210626_2124'),
    ]

    operations = [
        migrations.AlterField(
            model_name='application',
            name='date',
            field=models.DateField(default=datetime.date(2021, 6, 27), null=True),
        ),
    ]
