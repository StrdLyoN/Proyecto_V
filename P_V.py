import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

games_df = pd.read_csv('./games.csv')

print('EN ESTRE PROYECTO ANALIZAREMOS LOS DATOS DE UNA BASE DE DATOS RELATIVA A VIDEOJUEGOS EN DIFERENTES REGIONES, Y VERMEOS LOS COMPORTAMIENTOS DE VENTAS, QUE INFLUYE EN EL AUMENTO O DISMINUCION DE ESTAS EN LAS REGIONES QUE TENGA LA BASE DE DATOS, ASI COM TAMBIEN COMPROBAREMOS HIPOTESIS SOBRE INFLUENCIAS EN DATOS.')
print()

print(games_df.head())
print()

print(games_df.info())
print()

print(games_df.describe())
print()

print('Verificar si hay valores duplicados y ausentes')
datos_duplicados = games_df.duplicated().sum()
print(f'Filas ducplicadas : {datos_duplicados}')
print()
datos_nulos = games_df.isna().sum()
print(f'Valores nulos : \n{datos_nulos}')
print()

print('TRANSFORMAREMOS EL NOMBRE DE LAS COLUMNAS A MINISCULAS')
print()
games_df.columns = games_df.columns.str.lower()
print(games_df.columns)
print()

print('LOS VALORES NULOS EN LAS COLUMNAS NAME, GENRE Y YEAR_OF_RELEASE SE CAMBIARAN A DESCRIPCIONES TALES COMO NOMBRE, GENERO Y DESCONOCIDO Y CON VALORES IGUAL A CERO PARA LOS AÑOS, ASI COMO EL CAMBIO DE ESA COLUMNA A ENTEROS.')
print()

games_df['name'].fillna('Nombre desconocido', inplace=True)
games_df['genre'].fillna('Genero desconocido', inplace=True)
games_df['year_of_release'] = games_df['year_of_release'].fillna(0).astype('int')

print('CONVERTIREMOS TBD EN NINGUN VALOR, ASI MISMO PASAREMOS ESA COLUMNA A UN VALOR FLOAT PARA PODER MANIPULARLA')
print()

print(games_df['user_score'].unique())
print()

games_df['user_score'].replace('tbd', np.nan, inplace=True)
games_df['user_score'] = games_df['user_score'].astype('float')
print(games_df['user_score'].isna().sum())
print()

print(games_df['user_score'].describe())
print()

print(games_df.info())
print()

print('PARA LA COLUMNA RATING CAMBIAREMOS SUS VALORES NULOS A VALORES DESCONOCIDOS.')
print()

games_df['rating'].fillna('Rating desconocido', inplace=True)
print(games_df.info())
print()

print('CALCULAREMOS LA VENTA TOTAL PARA CADA JUEGO EN UNA COLUMNA SEPARADA')
print()

games_df['total_sales'] = games_df[['na_sales', 'eu_sales', 'jp_sales','other_sales']].sum(axis=1)
print(games_df.head())
print()

best_games_5 = games_df.nlargest(5, ['total_sales'])
print(best_games_5)
print()

print('MOSTRAREMOS LASA VENTAS POR AÑO')
print()

games_per_year = games_df.groupby('year_of_release')['name'].count().sort_values().reset_index()
games_per_year.columns = ['Año_de_lanzamiento','Cantidad_de_juegos']
games_per_year.query('Año_de_lanzamiento !=0', inplace=True)

plt.vlines(
x = games_per_year.Año_de_lanzamiento,
ymin = 0,
ymax = games_per_year.Cantidad_de_juegos,
alpha = 0.5,
linewidth = 10,
color = 'red'
)
plt.title('Juegos por año')
plt.xlabel('Años')
plt.ylabel('Ventas')
plt.show()

sales_per_platform = games_df.groupby('platform')['total_sales'].sum().sort_values(ascending=False).reset_index()
print(sales_per_platform.head())
print()

print('A CONTINUACION DESAROLLAREMOS UN CODIGO PARA QUE EN EL DIAGRAMA NOS MUESTRE LA VARIACION DE LAS DIFERENTES PLATAFORMAS POR LA MEDIA DE VENTAS, Y EN VERDE LAS QUE SON MASS RENTABLES QUE LAS QUE ESTARAN EN ROJO.')
print()

sales_per_platform['z_score'] = (sales_per_platform['total_sales'] - sales_per_platform['total_sales'].mean()) / sales_per_platform['total_sales'].std()
sales_per_platform['color'] = ['red' if puntaje <= 0 else 'green' for puntaje in sales_per_platform['z_score']]

plt.hlines(
y = sales_per_platform.platform,
xmin = 0,
xmax = sales_per_platform.z_score,
color = sales_per_platform.color,
linewidth = 10,
alpha = 0.5
)
plt.title('Ventas por plataforma')
plt.show()

print('LO QUE PODEMOS OBSERVAR ES QUE LAS PLATAFORMAS DE PS4, GBA, PS, DS, PS3,, WII, X360 Y PS2 SE MUESTRAN COMO LAS EMPRESAS RENTABLES DE TODA LA BASE DE DATOS A TRAVES DE LOS AÑOS. SIENDO PS2 LA MAS RENTABLE HASTA AHORA.')
print()

print('ANALIZAREMOS LA VIDA DE LAS PLATAFORMAS')
print()

life_time_platform = pd.pivot_table(
games_df,
index = 'year_of_release',
columns = 'platform',
values = 'total_sales',
aggfunc = 'sum')

life_time_platform.query('year_of_release > 0', inplace=True)
print(life_time_platform)
print()

filtered_data = life_time_platform.loc[2010:2016]
sns.set_theme(rc = {'figure.figsize':(15,10)})
sns.lineplot(
data = filtered_data
)
plt.show()

print('DE LO GRAFICADO PODEMOS OBSERVAR QUE A TRAVES DEL TIEMPO EMPRESAS OBTIENEN SUS MAXIMAS VENTA SY LUEGO BAJAN CONSIDERABLEMENTE, PERO HAY EMPRESAS QUE SE MANTIENEN CON POCAS VENTAS Y POR TENDENCIA OBTIENEN VENTAS GRANDES Y LUEGO VUELVEN A CAER PERO SE MANTIENEN.')
print()

print('A CONTINUACION GRAFICAREMOS UN DIAGRAMA DE CAJA PARA LAS VENTAS TOTALES DE CADA PLATAFORMA.')
print()

sns.boxplot(
x = 'platform',
y = 'total_sales',
data = games_df,
)
plt.title('Diagrama de caja de ventas totales')
plt.xlabel('Plataforma')
plt.ylabel('Ventas totales')
plt.show()

print('PODEMOS OBSERVAR QUE LAS PLATAFORMAS TIENEN ALGO EN COMUN, TIENEN CLIENTES QUE SOBREPASAN LA MEDIA Y MANTIENEN UN RANGO ENTRE VALORES MAXIMOS Y LA MEDIA, POR LO QUE ESOS CLIENTES DEBEN DE SER A LOS QUE MAS SE LES DEBE DE ESTUDIAR, PARA JUSTO AUMENTAR MAS CONSUMO, YA QUE SON LOS QUE ESTÁN POR ENCIMA DE TODOS LOS CONSUMIDORES.')
print()

print('AHORA ANALIZAREMOS ESPECIFICAMENTE LA PLATAFORMA PS2, VENTAS EN LAS DIFERENTES ZONAS, SU CORRELACION ENTRE LAS RESEÑAS Y SUS VENTAS TOTALES. ESTO LO VEREMOS REPRESENTADO POR UN DIAGRAMA DE DISPERSION.')
print()

platform_choosen = 'PS2'
platform_df = games_df[games_df['platform'] == platform_choosen]
platform_df = platform_df.dropna(subset=['user_score', 'total_sales'])

sns.scatterplot(
x = 'user_score',
y = 'total_sales',
data = platform_df
)
plt.title('Grafico de dispersion')
plt.xlabel('Reseñas de usuarios')
plt.ylabel('Ventas totales')
plt.grid(True)
plt.show()

print('OBSERVAMOS QUE ENTRE MAS ALTO EL VALOR DE LA RESEÑA MAS JUEGOS SE VENDEN A EXCEPCION DE UN PAR')
print()

correlacion = platform_df['user_score'].corr(platform_df['total_sales'])
print(f'La correlacion entre las reseñas de usuarios y las ventas totales en {platform_choosen} es: {correlacion:.2f}')
print()

sales_per_genre = games_df.groupby('genre')['total_sales'].sum().reset_index()
sales_per_genre = sales_per_genre.sort_values(by = 'total_sales', ascending = False)

sns.barplot(
data = sales_per_genre,
x = 'genre',
y = 'total_sales'
)
plt.title('Ventas por genero')
plt.xlabel('Genero')
plt.ylabel('Ventas totales')
plt.xticks(rotation = 45)
plt.show()

print('PODEMOS OBSERVAR DE ESTA GRÁFICA QUE EL GENERO MAS VENDIDO ES DE ACCION Y EL MAS BAJO ES DE ESTRATEGIA, AL IGUAL TENEMOS INFORMACION FALTANTE YA QUE NO EN TODOS LOS JUEGOS PUSIERON EL GENERO AL QUE PERTENECIAN.')
print()

print('ANALIZAREMOS LOS 5 PRINCIPALES PLATAFORMAS DE LAS REGIONES NA, EU Y JP.')
print()

top_na_platform = games_df.groupby('platform')['na_sales'].sum().nlargest(5).reset_index()
top_eu_platform = games_df.groupby('platform')['eu_sales'].sum().nlargest(5).reset_index()
top_jp_platform = games_df.groupby('platform')['jp_sales'].sum().nlargest(5).reset_index()
print(top_na_platform)
print()

print(top_eu_platform)
print()

print(top_jp_platform)
print()

print('OBSERVAMOS QUE PS2 ESTA PRESENTE EN LAS TRES REGIONES COMO UNA DE LAS 5 PLATAFORMAS LIDER, LO CUAL CORROBORA NUESTRO ANALISIS QUE HICIMOS CON ANTERIORIDAD QUE PS2 ES LA MAS RENTABLE.')
print()

total_na_sales = games_df['na_sales'].sum()
total_eu_sales = games_df['eu_sales'].sum()
total_jp_sales = games_df['jp_sales'].sum()

print(total_na_sales)
print()

print(total_eu_sales)
print()

print(total_jp_sales)
print()

print('NOS DAMOS CUENTA QUE EN LA REGION DE NA, LAS VENTAS SON MAYORES CASI DOBLANDO A LA REGION DE EU Y CASI CUADRIPLICANDO A LA REGION DE JP, ENTONCES EN NA SE CENTRA NUESTRO MERCADO QUE MAS IMPACTA.')
print()

print('ANALIZAREMOS COMO ES EL COMPORTAMIENTO DE LAS VENTAS CON LA CANTIDAD DE RATING QUE HAN TENIDO')
print()

rating_na_sales = games_df.groupby(['platform', 'rating'])['na_sales'].sum().reset_index()
print(rating_na_sales)
print()

sns.scatterplot(
x = 'rating',
y = 'na_sales',
color = 'red',
data = rating_na_sales
)
plt.title('Grafico de dispersion NA')
plt.xlabel('Valoracion de juegos')
plt.ylabel('Ventas totales')
plt.grid(True)
plt.show()

print('OBSERVAMOS QUE A PESAR DE QUE VARIOS JUEGOS NO TIENEN RATING LA GENTE LOS COMPRA Y ES UNA DE LOS 5 CATEGORIAS DE LAS QUE MAS VENTAS GENERA AL MENOS EN NA.')
print()

rating_eu_sales = games_df.groupby(['platform', 'rating'])['eu_sales'].sum().reset_index()
print(rating_eu_sales)
print()

sns.scatterplot(
x = 'rating',
y = 'eu_sales',
color = 'green',
data = rating_eu_sales
)
plt.title('Grafico de dispersion EU')
plt.xlabel('Valoracion de juegos')
plt.ylabel('Ventas totales')
plt.grid(True)
plt.show()

print('OBSERVAMOS QUE AL IGUAL QUE EN LA REGION NA, LOS QUE NO CUENTAN CON RATING ESTAN DENTRO DE LOS 5 MAS VENDIDOS')
print()

rating_jp_sales = games_df.groupby(['platform', 'rating'])['jp_sales'].sum().reset_index()
print(rating_jp_sales)
print()

sns.scatterplot(
x = 'rating',
y = 'jp_sales',
color = 'orange',
data = rating_jp_sales
)
plt.title('Grafico de dispersion JP')
plt.xlabel('Valoracion de juegos')
plt.ylabel('Ventas totales')
plt.grid(True)
plt.show()

print('EN ESTA REGION VEMOS QUE LOS JUEGOS SIN RATING TIENEN LAS MAYORES VENTAS.')
print()

platform_H_01 = games_df[games_df['platform'].isin(['XOne','PC'])]
mean_score_H_01 = platform_H_01.groupby('platform')['user_score'].mean()
print(mean_score_H_01)
print()

xbox_one_scores = games_df[games_df['platform'] == 'XOne']['user_score'].dropna()
pc_scores = games_df[games_df['platform'] == 'PC']['user_score'].dropna()
t_stat, p_value_platforms = stats.ttest_ind(xbox_one_scores, pc_scores)
print('Plataformas - Estadístico t:', t_stat, ', p-valor:', p_value_platforms)
print()

action_scores = games_df[games_df['genre'] == 'Action']['user_score'].dropna()
sports_scores = games_df[games_df['genre'] == 'Sports']['user_score'].dropna()
t_stat_genre, p_value_genre = stats.ttest_ind(action_scores, sports_scores)
print('Géneros - Estadístico t:', t_stat_genre, ', p-valor:', p_value_genre)
print()

alpha = 0.05
print(f'Valor alfa: {alpha}')
print()

if p_value_platforms < alpha:
    print("Rechazamos H0 para plataformas: Las calificaciones promedio son diferentes.")
else:
    print("No se puede rechazar H0 para plataformas: No hay evidencia suficiente para afirmar que las calificaciones promedio son diferentes.")

if p_value_genre < alpha:
    print("Rechazamos H0 para géneros: Las calificaciones promedio son diferentes.")
else:
    print("No se puede rechazar H0 para géneros: No hay evidencia suficiente para afirmar que las calificaciones promedio son diferentes.")
print()

xbox_one_data = games_df[games_df['platform'] == 'XOne']
pc_data = games_df[games_df['platform'] == 'PC']

levene_stat, levene_p = stats.levene(xbox_one_data['user_score'].dropna(), pc_data['user_score'].dropna())

print(f'Estadística de Levene: {levene_stat}, p-valor: {levene_p}')

equal_var = levene_p > 0.05

t_stat, t_p = stats.ttest_ind(
    xbox_one_data['user_score'].dropna(),
    pc_data['user_score'].dropna(),
    equal_var=equal_var)

print(f'Estadística t: {t_stat}, p-valor: {t_p}')

if t_p < 0.05:
    print("Rechazamos la hipótesis nula: Las calificaciones promedio son diferentes.")
else:
    print("No rechazamos la hipótesis nula: Las calificaciones promedio son las mismas.")
print()

action_data = games_df[games_df['genre'] == 'Action']
sports_data = games_df[games_df['genre'] == 'Sports']

levene_stat_genre, levene_p_genre = stats.levene(action_data['user_score'].dropna(),
    sports_data['user_score'].dropna())

print(f'Estadística de Levene (géneros): {levene_stat_genre}, p-valor: {levene_p_genre}')

equal_var_genre = levene_p_genre > 0.05

t_stat_genre, t_p_genre = stats.ttest_ind(action_data['user_score'].dropna(),
    sports_data['user_score'].dropna(),
    equal_var=equal_var_genre)

print(f'Estadística t (géneros): {t_stat_genre}, p-valor: {t_p_genre}')

if t_p_genre < 0.05:
    print("Rechazamos la hipótesis nula: Las calificaciones promedio entre Acción y Deportes son diferentes.")
else:
    print("No rechazamos la hipótesis nula: Las calificaciones promedio entre Acción y Deportes son las mismas.")
print()

print('SE REALIZO UNA PRUEBA T PARA DOS MUESTRAS INDEPENDIENTES PARA DETERMINAR SI HAY UNA DIFERENCIA SIGNIFICATIVA ENTRE LAS MEDIAS DE LAS DOS MUESTRAS, EL VALOR DE ALPHA, SE ESCOGIO COMO UN PROMEDIO QUE SIEMPRE SE TOMA POR LO GENERAL, QUE ES IGUAL AL VALOR 0.5')
print()
print('A LO LARGO DEL DESAROLLO Y MANIPULACION DE LOS DATOS POR ANALIZAR, NOS DIMOS CUENTA SOBRE LA VARIACION DE VENTAS EN LAS DIFERENTES REGIONES, TOMANDO EN CUENTA EVALUACIONES POR USUARIOS Y CRITICOS, VIMOS COMO ESTAS INFLUENCIABAN EN SUS VENTAS, SI HABIA UNA RELACION EN ESTAS. EN LO PERSONAL DEBEMOS DE PRESTAR MAS ATENCION EN LAS QUE NO ESTAN CLASIFICADAS CON ALGUN RATING, YA QUE PRESENTAN UNOS DE LOS 5 INGRESOS MAS GRANDES EN LAS TRES REGIONES DE VENTAS. TAMBIEN SE DEBE DE TENER EN CUENTA QUE MUCHOS USUARIOS SOBREPASAN LAS VENTAS PROMEDIO DE LAS DIFERENTES PLATAFORMAS, A LOS CUALES HAY QUE DEDICAR MAS ')
print()