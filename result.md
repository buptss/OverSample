/Library/Python/2.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
  'precision', 'predicted', average, warn_for)
/Library/Python/2.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
  'precision', 'predicted', average, warn_for)
\multirow{6}{*}{\textbf{ecoli}}
&ADASYN& 0.700& 0.636& 0.667& 0.781\\
&No Sample& 0.333& 0.091& 0.143& 0.297\\
&SMOTE& 0.625& 0.455& 0.526& 0.660\\
&SMOTEBorderline-1& 0.571& 0.364& 0.444& 0.591\\
&SMOTEBorderline-2& \textbf{0.727}& \textbf{0.727}& \textbf{0.727}& \textbf{0.835}\\
&SVMSMOTE& 0.500& 0.273& 0.353& 0.511\\
&random& 0.333& 0.091& 0.143& 0.297\\
\hline
\multirow{6}{*}{\textbf{optical_digits}}
&ADASYN& \textbf{0.981}& 0.912& 0.945& 0.954\\
&No Sample& 0.980& 0.860& 0.916& 0.926\\
&SMOTE& 0.964& 0.939& \textbf{0.951}& 0.967\\
&SMOTEBorderline-1& 0.980& 0.877& 0.926& 0.936\\
&SMOTEBorderline-2& 0.962& 0.877& 0.917& 0.935\\
&SVMSMOTE& 0.946& 0.930& 0.938& 0.962\\
&random& 0.925& \textbf{0.974}& 0.949& \textbf{0.983}\\
\hline
\multirow{6}{*}{\textbf{satimage}}
&ADASYN& 0.519& 0.824& 0.637& 0.867\\
&No Sample& \textbf{0.746}& 0.570& 0.646& 0.746\\
&SMOTE& 0.561& 0.758& 0.644& 0.840\\
&SMOTEBorderline-1& 0.563& 0.836& \textbf{0.673}& 0.880\\
&SMOTEBorderline-2& 0.516& \textbf{0.867}& 0.647& \textbf{0.887}\\
&SVMSMOTE& 0.556& 0.836& 0.668& 0.879\\
&random& 0.496& 0.842& 0.625& 0.872\\
\hline
\multirow{6}{*}{\textbf{pen_digits}}
&ADASYN& 0.973& 0.980& 0.976& 0.989\\
&No Sample& \textbf{1.000}& 0.980& 0.990& 0.990\\
&SMOTE& 0.996& \textbf{0.992}& \textbf{0.994}& \textbf{0.996}\\
&SMOTEBorderline-1& \textbf{1.000}& 0.965& 0.982& 0.982\\
&SMOTEBorderline-2& 0.992& 0.969& 0.980& 0.984\\
&SVMSMOTE& 0.992& 0.988& 0.990& 0.994\\
&random& 0.992& \textbf{0.992}& 0.992& 0.996\\
\hline
\multirow{6}{*}{\textbf{abalone}}
&ADASYN& 0.307& 0.766& 0.438& 0.780\\
&No Sample& \textbf{0.800}& 0.036& 0.069& 0.190\\
&SMOTE& 0.321& 0.721& 0.444& 0.768\\
&SMOTEBorderline-1& 0.325& 0.586& 0.418& 0.708\\
&SMOTEBorderline-2& 0.331& 0.541& 0.411& 0.686\\
&SVMSMOTE& 0.312& 0.523& 0.391& 0.672\\
&random& 0.311& \textbf{0.820}& \textbf{0.450}& \textbf{0.802}\\
\hline
\multirow{6}{*}{\textbf{sick_euthyroid}}
&ADASYN& 0.824& 0.893& 0.857& 0.934\\
&No Sample& 0.860& 0.881& 0.871& 0.931\\
&SMOTE& 0.824& 0.893& 0.857& 0.934\\
&SMOTEBorderline-1& 0.841& 0.881& 0.860& 0.929\\
&SMOTEBorderline-2& 0.796& 0.881& 0.836& 0.926\\
&SVMSMOTE& \textbf{0.874}& 0.905& \textbf{0.889}& 0.944\\
&random& 0.808& \textbf{0.952}& 0.874& \textbf{0.963}\\
\hline
\multirow{6}{*}{\textbf{spectrometer}}
&ADASYN& \textbf{0.846}& \textbf{0.846}& \textbf{0.846}& \textbf{0.912}\\
&No Sample& 0.833& 0.385& 0.526& 0.618\\
&SMOTE& 0.714& 0.769& 0.741& 0.862\\
&SMOTEBorderline-1& 0.700& 0.538& 0.609& 0.725\\
&SMOTEBorderline-2& 0.727& 0.615& 0.667& 0.775\\
&SVMSMOTE& 0.750& 0.692& 0.720& 0.822\\
&random& 0.700& 0.538& 0.609& 0.725\\
\hline
\multirow{6}{*}{\textbf{car_eval_34}}
&ADASYN& \textbf{0.951}& \textbf{1.000}& \textbf{0.975}& \textbf{0.997}\\
&No Sample& \textbf{0.951}& \textbf{1.000}& \textbf{0.975}& \textbf{0.997}\\
&SMOTE& \textbf{0.951}& \textbf{1.000}& \textbf{0.975}& \textbf{0.997}\\
&SMOTEBorderline-1& 0.929& \textbf{1.000}& 0.963& 0.996\\
&SMOTEBorderline-2& \textbf{0.951}& \textbf{1.000}& \textbf{0.975}& \textbf{0.997}\\
&SVMSMOTE& 0.830& \textbf{1.000}& 0.907& 0.990\\
&random& 0.848& \textbf{1.000}& 0.918& 0.991\\
\hline
\multirow{6}{*}{\textbf{isolet}}
&ADASYN& 0.704& 0.902& 0.791& 0.936\\
&No Sample& \textbf{0.886}& 0.826& \textbf{0.855}& 0.905\\
&SMOTE& 0.759& 0.955& 0.846& 0.966\\
&SMOTEBorderline-1& 0.753& 0.902& 0.821& 0.939\\
&SMOTEBorderline-2& 0.702& 0.909& 0.792& 0.940\\
&SVMSMOTE& 0.678& 0.939& 0.787& 0.953\\
&random& 0.753& \textbf{0.970}& 0.848& \textbf{0.973}\\
\hline
\multirow{6}{*}{\textbf{us_crime}}
&ADASYN& 0.463& 0.528& 0.494& 0.709\\
&No Sample& \textbf{0.720}& 0.500& \textbf{0.590}& 0.702\\
&SMOTE& 0.476& 0.556& 0.513& 0.727\\
&SMOTEBorderline-1& 0.477& 0.583& 0.525& 0.745\\
&SMOTEBorderline-2& 0.479& \textbf{0.639}& 0.548& \textbf{0.777}\\
&SVMSMOTE& 0.467& 0.583& 0.519& 0.744\\
&random& 0.429& 0.583& 0.494& 0.740\\
\hline
\multirow{6}{*}{\textbf{yeast_ml8}}
&ADASYN& 0.098& 0.128& 0.111& 0.343\\
&No Sample& 0.000& 0.000& 0.000& 0.000\\
&SMOTE& 0.099& \textbf{0.179}& 0.127& \textbf{0.399}\\
&SMOTEBorderline-1& 0.077& 0.077& 0.077& 0.268\\
&SMOTEBorderline-2& 0.136& 0.154& \textbf{0.145}& 0.379\\
&SVMSMOTE& 0.083& 0.077& 0.080& 0.269\\
&random& \textbf{0.161}& 0.128& 0.143& 0.350\\
\hline
\multirow{6}{*}{\textbf{scene}}
&ADASYN& 0.246& \textbf{0.390}& \textbf{0.302}& \textbf{0.597}\\
&No Sample& \textbf{0.571}& 0.098& 0.167& 0.312\\
&SMOTE& 0.225& \textbf{0.390}& 0.286& 0.593\\
&SMOTEBorderline-1& 0.271& 0.317& 0.292& 0.545\\
&SMOTEBorderline-2& 0.224& 0.366& 0.278& 0.576\\
&SVMSMOTE& 0.286& 0.293& 0.289& 0.526\\
&random& 0.323& 0.244& 0.278& 0.485\\
\hline
\multirow{6}{*}{\textbf{libras_move}}
&ADASYN& 0.875& \textbf{0.875}& 0.875& 0.930\\
&No Sample& \textbf{1.000}& 0.750& 0.857& 0.866\\
&SMOTE& 0.875& \textbf{0.875}& 0.875& 0.930\\
&SMOTEBorderline-1& 0.875& \textbf{0.875}& 0.875& 0.930\\
&SMOTEBorderline-2& \textbf{1.000}& \textbf{0.875}& \textbf{0.933}& \textbf{0.935}\\
&SVMSMOTE& \textbf{1.000}& \textbf{0.875}& \textbf{0.933}& \textbf{0.935}\\
&random& \textbf{1.000}& \textbf{0.875}& \textbf{0.933}& \textbf{0.935}\\
\hline
\multirow{6}{*}{\textbf{thyroid_sick}}
&ADASYN& 0.857& 0.964& \textbf{0.908}& 0.977\\
&No Sample& \textbf{0.902}& 0.821& 0.860& 0.904\\
&SMOTE& 0.867& 0.929& 0.897& 0.959\\
&SMOTEBorderline-1& 0.833& \textbf{0.982}& 0.902& \textbf{0.985}\\
&SMOTEBorderline-2& 0.823& 0.911& 0.864& 0.948\\
&SVMSMOTE& 0.852& 0.929& 0.889& 0.959\\
&random& 0.797& \textbf{0.982}& 0.880& 0.983\\
\hline
\multirow{6}{*}{\textbf{coil_2000}}
&ADASYN& 0.302& 0.081& 0.127& 0.282\\
&No Sample& \textbf{1.000}& 0.012& 0.025& 0.111\\
&SMOTE& 0.288& 0.093& 0.141& 0.303\\
&SMOTEBorderline-1& 0.274& 0.143& 0.188& 0.373\\
&SMOTEBorderline-2& 0.348& 0.099& 0.155& 0.313\\
&SVMSMOTE& 0.353& 0.149& 0.210& 0.382\\
&random& 0.152& \textbf{0.609}& \textbf{0.243}& \textbf{0.681}\\
\hline
\multirow{6}{*}{\textbf{arrhythmia}}
&ADASYN& 0.667& \textbf{1.000}& 0.800& 0.991\\
&No Sample& 0.500& 0.500& 0.500& 0.701\\
&SMOTE& 0.800& \textbf{1.000}& \textbf{0.889}& \textbf{0.995}\\
&SMOTEBorderline-1& 0.800& \textbf{1.000}& \textbf{0.889}& \textbf{0.995}\\
&SMOTEBorderline-2& \textbf{1.000}& 0.750& 0.857& 0.866\\
&SVMSMOTE& 0.667& \textbf{1.000}& 0.800& 0.991\\
&random& 0.667& \textbf{1.000}& 0.800& 0.991\\
\hline
\multirow{6}{*}{\textbf{solar_flare_m0}}
&ADASYN& 0.300& 0.273& 0.286& 0.517\\
&No Sample& 0.333& 0.182& 0.235& 0.424\\
&SMOTE& 0.300& 0.273& 0.286& 0.517\\
&SMOTEBorderline-1& 0.333& 0.364& \textbf{0.348}& 0.596\\
&SMOTEBorderline-2& \textbf{0.429}& 0.273& 0.333& 0.519\\
&SVMSMOTE& 0.333& 0.273& 0.300& 0.518\\
&random& 0.096& \textbf{0.727}& 0.170& \textbf{0.752}\\
\hline
\multirow{6}{*}{\textbf{oil}}
&ADASYN& 0.750& 0.643& 0.692& 0.796\\
&No Sample& \textbf{1.000}& 0.286& 0.444& 0.535\\
&SMOTE& 0.692& 0.643& 0.667& 0.794\\
&SMOTEBorderline-1& 0.800& 0.571& 0.667& 0.753\\
&SMOTEBorderline-2& 0.833& \textbf{0.714}& \textbf{0.769}& \textbf{0.841}\\
&SVMSMOTE& 0.727& 0.571& 0.640& 0.751\\
&random& 0.727& 0.571& 0.640& 0.751\\
\hline
\multirow{6}{*}{\textbf{car_eval_4}}
&ADASYN& 0.895& \textbf{1.000}& 0.944& 0.998\\
&No Sample& \textbf{0.944}& \textbf{1.000}& \textbf{0.971}& \textbf{0.999}\\
&SMOTE& 0.895& \textbf{1.000}& 0.944& 0.998\\
&SMOTEBorderline-1& 0.895& \textbf{1.000}& 0.944& 0.998\\
&SMOTEBorderline-2& 0.895& \textbf{1.000}& 0.944& 0.998\\
&SVMSMOTE& 0.850& \textbf{1.000}& 0.919& 0.996\\
&random& 0.895& \textbf{1.000}& 0.944& 0.998\\
\hline
\multirow{6}{*}{\textbf{wine_quality}}
&ADASYN& 0.216& \textbf{0.533}& 0.308& 0.703\\
&No Sample& \textbf{0.833}& 0.111& 0.196& 0.333\\
&SMOTE& 0.220& \textbf{0.533}& 0.312& \textbf{0.704}\\
&SMOTEBorderline-1& 0.213& 0.422& 0.284& 0.630\\
&SMOTEBorderline-2& 0.202& 0.511& 0.289& 0.687\\
&SVMSMOTE& 0.328& 0.444& \textbf{0.377}& 0.655\\
&random& 0.193& 0.467& 0.273& 0.657\\
\hline
\multirow{6}{*}{\textbf{letter_img}}
&ADASYN& 0.871& 0.985& 0.924& 0.989\\
&No Sample& \textbf{0.978}& 0.909& 0.942& 0.953\\
&SMOTE& 0.873& 0.970& 0.919& 0.982\\
&SMOTEBorderline-1& 0.949& 0.944& \textbf{0.947}& 0.971\\
&SMOTEBorderline-2& 0.853& 0.848& 0.851& 0.918\\
&SVMSMOTE& 0.848& 0.990& 0.914& 0.991\\
&random& 0.764& \textbf{1.000}& 0.867& \textbf{0.994}\\
\