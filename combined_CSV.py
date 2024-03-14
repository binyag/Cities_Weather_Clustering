# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:40:36 2024

@author: USER
"""

import csv
import os

# הגדרת שם הקובץ המאוחד
output_file = "combined_data.csv"

# פתיחת קובץ חדש לכתיבה
with open(output_file, 'w',encoding='utf-8' , newline='') as csvfile:
    writer = csv.writer(csvfile)

    # הליכה על כל הקבצים בתיקייה
    for filename in os.listdir("C:\\Users\\USER\\OneDrive - Ariel University\\לימודים תעונ\\שנה ד\\נושאים מתקדמים בלמידת מכונה\\פרויקט גמר מזא"):
        if filename.endswith(".csv"):
            # פתיחת קובץ CSV לקריאה
            with open(os.path.join("C:\\Users\\USER\\OneDrive - Ariel University\\לימודים תעונ\\שנה ד\\נושאים מתקדמים בלמידת מכונה\\פרויקט גמר מזא", filename), 'r',encoding='utf-8', newline='') as f:
                reader = csv.reader(f)

                # קריאה שורה אחר שורה
                for row in reader:
                    # כתיבת השורה לקובץ המאוחד
                    writer.writerow(row)

print("קבצי CSV אוחדו בהצלחה!")