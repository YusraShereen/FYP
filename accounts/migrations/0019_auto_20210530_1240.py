# Generated by Django 3.2.3 on 2021-05-30 12:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0018_delete_hotel'),
    ]

    operations = [
        migrations.CreateModel(
            name='uploadDoc',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('cnic', models.CharField(max_length=50, null=True)),
                ('Main_Img', models.ImageField(upload_to='images/')),
            ],
        ),
        migrations.DeleteModel(
            name='custDoc',
        ),
    ]