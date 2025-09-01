#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Візуальне відображення пауз у фонетичній транскрипції.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from utilites import (
    phonemes_with_pauses_to_ipa,
    analyze_pause_structure,
    PAUSE_SYMBOLS
)

def visualize_pause_timeline(phonemes_with_pauses, title="Часова лінія пауз"):
    """
    Візуалізує паузи у вигляді часової лінії.
    
    Args:
        phonemes_with_pauses: список фонем з паузами
        title: заголовок графіку
    """
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Кольори для різних типів пауз
    pause_colors = {
        "SP": "#FFE6E6",         # світло-рожевий (коротка пауза)
        "SP2": "#FFB3B3",        # рожевий (середня пауза)
        "SIL": "#FF8080",        # темно-рожевий (довга пауза)
        "WORD_BOUNDARY": "#E6F3FF",  # світло-блакитний
    }
    
    # Висота блоків
    phoneme_height = 0.8
    pause_height = 0.6
    
    x_pos = 0
    max_y = 1
    
    for i, token in enumerate(phonemes_with_pauses):
        if token in PAUSE_SYMBOLS:
            # Відображаємо паузу
            color = pause_colors.get(token, "#CCCCCC")
            width = 0.5 if token == "SP" else (1.0 if token == "SP2" else 1.5)
            
            rect = Rectangle((x_pos, 0.2), width, pause_height, 
                           facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Додаємо підпис паузи
            ax.text(x_pos + width/2, 0.5, PAUSE_SYMBOLS[token], 
                   ha='center', va='center', fontsize=12, fontweight='bold')
            
            x_pos += width
        else:
            # Відображаємо фонему
            rect = Rectangle((x_pos, 0.1), 0.8, phoneme_height, 
                           facecolor='lightblue', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Додаємо підпис фонеми
            ax.text(x_pos + 0.4, 0.5, token, 
                   ha='center', va='center', fontsize=8, rotation=45)
            
            x_pos += 0.8
    
    ax.set_xlim(0, x_pos)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('Позиція в послідовності')
    ax.set_title(title)
    ax.set_yticks([])
    
    # Легенда
    legend_elements = []
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Фонема'))
    for pause_type, color in pause_colors.items():
        if pause_type != "WORD_BOUNDARY":  # Не показуємо межі слів в легенді
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                               label=pause_type.replace('_', ' ').title()))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig

def plot_pause_statistics(phonemes_with_pauses, title="Статистика пауз"):
    """
    Створює діаграму статистики пауз.
    
    Args:
        phonemes_with_pauses: список фонем з паузами
        title: заголовок графіку
    """
    analysis = analyze_pause_structure(phonemes_with_pauses)
    
    # Підготовка даних для діаграми
    pause_types = []
    counts = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    for pause_type, count in analysis['pause_counts'].items():
        if count > 0:
            pause_types.append(pause_type.replace('_', ' ').title())
            counts.append(count)
    
    if not counts:
        print("Немає пауз для відображення")
        return None
    
    # Створення кругової діаграми
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Кругова діаграма
    wedges, texts, autotexts = ax1.pie(counts, labels=pause_types, autopct='%1.1f%%',
                                      colors=colors[:len(counts)], startangle=90)
    ax1.set_title(f'{title}\n(Всього пауз: {analysis["total_pauses"]})')
    
    # Стовпчаста діаграма
    bars = ax2.bar(pause_types, counts, color=colors[:len(counts)])
    ax2.set_title('Кількість пауз по типах')
    ax2.set_ylabel('Кількість')
    ax2.tick_params(axis='x', rotation=45)
    
    # Додаємо значення на стовпчики
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_phoneme_duration_model(phonemes_with_pauses, title="Модель тривалості фонем"):
    """
    Моделює та візуалізує тривалість фонем з урахуванням пауз.
    
    Args:
        phonemes_with_pauses: список фонем з паузами
        title: заголовок графіку
    """
    # Базові тривалості (в умовних одиницях)
    base_durations = {
        # Короткі голосні
        "IH": 0.08, "UH": 0.08, "AH": 0.09,
        # Довгі голосні
        "IY": 0.12, "UW": 0.12, "AA": 0.13, "AO": 0.13,
        # Дифтонги
        "AY": 0.15, "AW": 0.15, "OY": 0.15, "EY": 0.14, "OW": 0.14,
        # Приголосні (середні значення)
        "P": 0.06, "B": 0.07, "T": 0.06, "D": 0.07, "K": 0.07, "G": 0.08,
        "F": 0.08, "V": 0.07, "S": 0.09, "Z": 0.08, "SH": 0.10, "ZH": 0.09,
        "M": 0.08, "N": 0.07, "NG": 0.08, "L": 0.08, "R": 0.08,
        "W": 0.06, "Y": 0.05, "HH": 0.05,
    }
    
    # Тривалості пауз
    pause_durations = {
        "SP": 0.15,              # коротка пауза
        "SP2": 0.30,             # середня пауза
        "SIL": 0.50,             # довга пауза
        "WORD_BOUNDARY": 0.05,   # межа слова
    }
    
    times = []
    durations = []
    labels = []
    colors = []
    current_time = 0
    
    for token in phonemes_with_pauses:
        if token in PAUSE_SYMBOLS:
            duration = pause_durations.get(token, 0.2)
            color = 'red'
        else:
            # Отримуємо базову фонему без наголосу
            base_phoneme = token.rstrip('012')
            duration = base_durations.get(base_phoneme, 0.08)
            color = 'blue'
        
        times.append(current_time)
        durations.append(duration)
        labels.append(token)
        colors.append(color)
        current_time += duration
    
    # Створення графіку
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Стовпчики для кожної фонеми/паузи
    bars = ax.bar(range(len(phonemes_with_pauses)), durations, color=colors, alpha=0.7)
    
    # Налаштування осей
    ax.set_xlabel('Позиція в послідовності')
    ax.set_ylabel('Тривалість (с)')
    ax.set_title(title)
    ax.set_xticks(range(len(phonemes_with_pauses)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Додаємо значення тривалості на стовпчики
    for i, (bar, duration) in enumerate(zip(bars, durations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{duration:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Легенда
    blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label='Фонеми')
    red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Паузи')
    ax.legend(handles=[blue_patch, red_patch])
    
    plt.tight_layout()
    return fig

def create_pause_visualization_demo():
    """Демонстрація всіх візуалізацій."""
    
    # Приклад даних
    sample_phonemes = [
        "HH", "EH1", "L", "OW0", "WORD_BOUNDARY",     # Hello
        "M", "AY1", "SP",                             # my
        "N", "EY1", "M", "SP2",                       # name
        "IH1", "Z", "SP",                             # is
        "JH", "AA1", "N", "SIL",                      # John
        "AY1", "M", "WORD_BOUNDARY",                  # I'm
        "F", "R", "AH1", "M", "SIL",                  # from
        "Y", "UW1", "K", "R", "EY1", "N"              # Ukraine
    ]
    
    print("Створення візуалізацій...")
    
    # 1. Часова лінія
    fig1 = visualize_pause_timeline(sample_phonemes, "Часова лінія: 'Hello my name is John'")
    plt.show()
    
    # 2. Статистика пауз
    fig2 = plot_pause_statistics(sample_phonemes, "Статистика пауз у реченні")
    plt.show()
    
    # 3. Модель тривалості
    fig3 = plot_phoneme_duration_model(sample_phonemes, "Модель тривалості фонем та пауз")
    plt.show()
    
    # IPA представлення
    ipa_result = phonemes_with_pauses_to_ipa(sample_phonemes)
    print(f"\nIPA транскрипція: {ipa_result}")
    
    # Аналіз
    analysis = analyze_pause_structure(sample_phonemes)
    print(f"Загальна статистика: {analysis['total_pauses']} пауз з {analysis['total_phonemes']} елементів")

if __name__ == "__main__":
    create_pause_visualization_demo()
