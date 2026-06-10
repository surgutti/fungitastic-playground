# Raport zmian eksperymentalnych

## 1. Cel zmian

Celem zmian było zwiększenie jakości segmentacji grzybów w projekcie FungiTastic przez:

- przejście z małych architektur i 300 px obrazów na silniejsze modele przy 512 px oraz opcjonalnie 640 px,
- dodanie lepszego opakowania treningowego z mocniejszą funkcją straty,
- wykorzystanie możliwości pojedynczego mocnego GPU przez `bf16-mixed`, większe modele i akumulację gradientu,
- przygotowanie powtarzalnego planu eksperymentów: baseline, smoke test, overfit-check, główny trening, TTA.

## 2. Zmiany w datasetach

Dodany został plik:

- `scripts/prepare_dataset_configurable.py`

Skrypt pozwala przygotować wersje datasetu w wyższej rozdzielczości, np. 512 px z obrazów 500p oraz 640 px z obrazów 720p. Zamiast sztywnego `TARGET_SIZE = 300`, można teraz sterować parametrami:

```bash
uv run python -m scripts.prepare_dataset_configurable \
  --subset m \
  --image_size 500 \
  --target_size 512 \
  --resize_mode letterbox
```

Najważniejsza zmiana to `letterbox`, czyli zachowanie proporcji obrazu i dopadanie do kwadratu. To jest bezpieczniejsze niż agresywne centralne cropowanie, bo zmniejsza ryzyko obcięcia kapelusza, trzonu albo rzadkich części typu pierścień.

Dodany został też:

- `scripts/dataset_stats.py`

Ten skrypt liczy częstości klas w maskach i wypisuje proponowane wagi do cross entropy. Z logów wynika, że klasy są bardzo niezbalansowane: tło dominuje, a `pores` i `ring` są rzadkie. To uzasadnia używanie ważonego CE, Dice oraz opcjonalnie Focal/Tversky loss.

## 3. Zmiany w modelach: ResNet U-Net

Dodany został plik:

- `src/models/architectures/resnet_unet.py`

Jest to architektura typu ResNet encoder + U-Net/FPN decoder. Wspierane backbone'y:

- `resnet50`,
- `resnet101`,
- `wide_resnet50_2`.

Encoder jest ImageNet-pretrained, więc model startuje z filtrami znającymi podstawowe krawędzie, tekstury i obiekty. To powinno pomagać szczególnie przy małym lub średnim zbiorze danych, gdzie trenowanie od zera łatwo daje gorszą generalizację.

Decoder ma:

- skip connections z wcześniejszych poziomów ResNeta,
- upsampling biliniowy,
- bloki konwolucyjne z BatchNorm i SiLU,
- ASPP w bottlenecku,
- lekką uwagę typu squeeze-excite.

Dlaczego to powinno poprawić wynik:

- skip connections pomagają odzyskać szczegóły przestrzenne potrzebne w segmentacji,
- ASPP zwiększa pole widzenia i pomaga widzieć części grzyba w kontekście całego owocnika,
- pretrained ResNet daje silniejszy encoder niż mały customowy encoder trenowany od zera,
- większa rozdzielczość pomaga szczególnie na małych klasach i granicach masek.

## 4. Zmiany w modelach: EncDecNet v2

Dodane zostały pliki:

- `src/models/architectures/encdecnet_v2.py`,
- `src/models/modules/residual_block.py`,
- `src/models/modules/attention_gate.py`,
- `src/config/encdecnet_v2_segmenter.py`.

`encdecnet_v2` jest mocniejszą wersją prostego EncDecNet. Zamiast zwykłych bloków konwolucyjnych używa bloków residualnych, ma głębszy encoder/decoder, bottleneck z dropoutem oraz attention gates na skip connectionach.

Dlaczego to powinno poprawić wynik:

- residual blocks ułatwiają trenowanie głębszego modelu i zmniejszają ryzyko zaniku gradientu,
- attention gates filtrują skip connectiony, więc decoder powinien dostawać mniej szumu z encodera,
- głębszy bottleneck daje większe pole widzenia i mocniejszą reprezentację semantyczną,
- jest to dobry eksperyment pośredni między małym modelem własnym a dużymi pretrained ResNetami.

## 5. Zmiany w module treningowym

Dodany został plik:

- `src/models/advanced_segmentation_model.py`

Nowy wrapper Lightning zachowuje prosty interfejs repozytorium: backbone zwraca embedding `[B, C, H, W]`, a wrapper dodaje segmentation head.

Dodane elementy:

- weighted cross entropy,
- foreground Dice loss,
- opcjonalny Focal loss,
- opcjonalny Tversky loss,
- per-class IoU logging,
- `val/mean_iou` liczony bez tła,
- AdamW z osobnymi grupami parametrów dla weight decay,
- scheduler cosine albo one-cycle.

Dlaczego to może poprawić wynik:

- samo pixel accuracy byłoby mylące, bo tło zajmuje większość pikseli,
- Dice premiuje poprawne pokrycie obszarów klas, a nie tylko klasyfikację pojedynczych pikseli,
- Focal/Tversky może pomóc rzadkim klasom, gdy CE i Dice nadal zbyt mocno ignorują `pores` i `ring`,
- per-class IoU pozwala zobaczyć, czy model naprawdę poprawia rzadkie klasy, a nie tylko tło/kapelusz.

## 6. Zmiany w treningu GPU

Dodany został plik:

- `scripts/train_model_gpu.py`

Jest to mocniejszy entrypoint treningowy niż podstawowy `scripts/train_model.py`. Obsługuje:

- `bf16-mixed`,
- GPU accelerator,
- gradient accumulation,
- gradient clipping,
- `fast_dev_run`,
- `overfit_batches`,
- opcjonalne `torch.compile` dla backbone'u,
- override liczby epoch,
- override liczby workerów DataLoadera.

Po analizie logów poprawiony został problem z `--overfit_batches`. Lightning rozróżnia `int` i `float`: `8` oznacza 8 batchy, a `0.02` oznacza 2% danych. Gdy Click parsował `8` jako `8.0`, overfit-check nie działał zgodnie z intencją. Dodałem parser, który zachowuje tę różnicę.

## 7. Konfiguracje eksperymentów

Dodane konfiguracje:

- `src/config/encdecnet_big_300.py`,
- `src/config/encdecnet_v2_segmenter.py`,
- `src/config/resnet50_unet_512.py`,
- `src/config/resnet101_unet_512.py`,
- `src/config/wide_resnet50_2_unet_512.py`,
- `src/config/resnet101_unet_640.py`.

Hierarchia eksperymentów:

1. baseline ze starego `encdecnet_segmenter.py`,
2. większy EncDecNet przy 300 px,
3. EncDecNet v2 z residual blocks i attention gates,
4. ResNet50 U-Net 512 px,
5. ResNet101 U-Net 512 px,
6. Wide-ResNet50-2 U-Net 512 px,
7. najlepszy model 512 px z TTA,
8. ResNet101 U-Net 640 px,
9. najlepszy model 640 px z TTA.

## 8. Test-time augmentation

Dodany został plik:

- `scripts/evaluate_tta.py`

Pozwala ewaluować checkpoint z:

- horizontal flip,
- multi-scale inference, np. `0.75,1.0,1.25`.

TTA nie zmienia modelu, ale uśrednia predykcje z kilku transformacji. Zwykle daje małą, ale stabilną poprawę końcowego wyniku, szczególnie gdy segmentowane obiekty mają różne skale.

## 9. Dokumentacja eksperymentów

Dodany został plik:

- `EXPERIMENTS.md`

Zawiera gotową kolejność uruchamiania:

- przygotowanie datasetu,
- smoke testy,
- overfit-check,
- główne eksperymenty,
- wariant 640 px,
- ewaluację TTA.

## 10. Wnioski z aktualnych logów

Z logów wynika, że:

- dataset 512 px został przygotowany poprawnie,
- zbiór treningowy ma 13803 obrazy,
- `val` ma 8672 obrazy,
- `test` ma 9765 obrazów,
- GPU CUDA jest wykrywane i używane,
- `bf16-mixed` jest aktywne,
- smoke testy dla ResNet50, ResNet101 i Wide-ResNet50-2 przechodzą,
- modele mają odpowiednio około 55.5M, 86.7M oraz 111M trenowalnych parametrów,
- pierwotny overfit-check uruchomił pełną epokę zamiast 8 batchy, dlatego został poprawiony parser parametru `--overfit_batches`.

## 11. Rekomendowany następny krok

Po `git pull` uruchomić:

```bash
uv run python -m scripts.train_model_gpu src/config/resnet50_unet_512.py \
  --overfit_batches 8 \
  --max_epochs 30 \
  --no_wandb
```

Jeżeli strata spada i model potrafi niemal zapamiętać 8 batchy, uruchomić pełny trening:

```bash
uv run python -m scripts.train_model_gpu src/config/resnet50_unet_512.py \
  --precision bf16-mixed
```

Dopiero po otrzymaniu stabilnego wyniku ResNet50 warto uruchamiać ResNet101 i Wide-ResNet50-2. Dzięki temu będzie wiadomo, czy większe modele faktycznie pomagają, czy tylko zwiększają koszt treningu.

## 12. Ryzyka

- Zbyt agresywne wagi klas mogą powodować nadpredykowanie rzadkich klas.
- 640 px może dać lepsze granice, ale też mocniej ogranicza batch size.
- TTA powinno być używane tylko do finalnej ewaluacji, nie do szybkiej iteracji.
- Jeżeli overfit-check nie zapamięta 8 batchy, trzeba sprawdzić alignment obrazów i masek, augmentacje oraz funkcję straty.
