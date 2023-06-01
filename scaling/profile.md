## model profiling

### 1. model list

| abbr.         | model                                          | source   |
|---------------|------------------------------------------------|----------|
| center_mobile | centernet_mobilenetv2_fpn_od                   | [...](.) |
| center_resnet | centernet_resnet50_v1_fpn_512x512_coco17_tpu-8 |          |
| ssd_mobilenet | ssd_mobilenet_v2_320x320_coco17_tpu-8          |          |
| nasnet        | NASNetLarge                                    |          |
| inception     | InceptionResNetV2                              |          |
| efficient     | EfficientNetB7                                 |          |

### 2. cold start (s)

| model name    | type           | load time | first infer |
|---------------|----------------|-----------|-------------|
| center_mobile | detection      | 2.9       | 2.7         |
| center_resnet | detection      | 2.6       | 3.5         |
| ssd_mobilenet | detection      | 3.4       | 8.9         |
| nasnet        | classification | 9         | 8.3         |
| inception     | classification | 5.5       | 4.6         |
| efficient     | classification | 7.3       | 7.1         |

### 3. duration (ms) and throughput (rps)


| model name    | batch size | first init | first AoT | duration | throughput |
|---------------|------------|------------|-----------|----------|------------|
| center_mobile | 1          | 243        | 19        | 15       | 66         |
|               | 2          | 267        | 23        | 17       | 120        |
|               | 4          | 283        | 25        | 22       | 180        |
|               | 8          | 331        | 42        | 38       | 208        |
|               | 16         | 439        | 73        | 69       | 232        |
|               | 32         | 662        | 135       | 130      | 246        |
| center_resnet | 1          | 449        | 32        | 28       | 36         |
|               | 2          | 522        | 36        | 33       | 60         |
|               | 4          | 628        | 47        | 47       | 84         |
|               | 8          | 684        | 76        | 75       | 106        |
|               | 16         | 1014       | 130       | 127      | 126        |
|               | 32         | 1709       | 233       | 236      | 135        |
| ssd_mobilenet | 1          | 330        | 75        | 69       | 15         |
|               | 2          | 388        | 114       | 117      | 17         |
|               | 4          | 502        | 200       | 209      | 19         |
|               | 8          | 679        | 398       | 396      | 20         |
|               | 16         | 1085       | 763       | 771      | 21         |
|               | 32         | 1906       | 1539      | 1544     | 21         |
| nasnet        | 1          | 258        | 51        | 45       | 22         |
|               | 2          | 298        | 55        | 52       | 39         |
|               | 4          | 334        | 78        | 73       | 55         |
|               | 8          | 418        | 117       | 118      | 68         |
|               | 16         | 584        | 208       | 232      | 69         |
|               | 32         | 916        | 381       | 388      | 82         |
| inception     | 1          | 410        | 37        | 34       | 29         |
|               | 2          | 430        | 42        | 39       | 51         |
|               | 4          | 452        | 47        | 48       | 84         |
|               | 8          | 487        | 70        | 68       | 117        |
|               | 16         | 607        | 103       | 106      | 151        |
|               | 32         | 881        | 190       | 193      | 166        |
| efficient     | 1          | 340        | 74        | 61       | 16         |
|               | 2          | 410        | 101       | 99       | 20         |
|               | 4          | 543        | 170       | 172      | 23         |
|               | 8          | 739        | 308       | 320      | 25         |
|               | 16         | 1234       | 626       | 637      | 25         |
|               | 32         | 2181       | 1216      | 1245     | 26         |

### 4. ...

| model name    | type           | load(s) | first(s) | b1/t | b2/t | b4/t | b8/t | b16/t | b32/t |
|---------------|----------------|---------|----------|------|------|------|------|-------|-------|
| center_mobile | detection      | 2.9     | 2.7      | 15   | 17   | 22   | 38   | 69    | 130   |
|               |                |         |          | 67   | 118  | 182  | 211  | 232   | 246   |
| center_resnet | detection      | 2.6     | 3.5      | 28   | 33   | 47   | 75   | 127   | 236   |
|               |                |         |          | 36   | 61   | 85   | 107  | 126   | 136   |
| ssd_mobilenet | detection      | 3.4     | 8.9      | 69   | 117  | 209  | 396  | 771   | 1544  |
|               |                |         |          | 14   | 17   | 19   | 20   | 21    | 21    |
| nasnet        | classification | 9.0     | 8.3      | 45   | 52   | 73   | 118  | 232   | 388   |
|               |                |         |          | 22   | 38   | 55   | 68   | 69    | 82    |
| inception     | classification | 5.5     | 4.6      | 34   | 39   | 48   | 68   | 106   | 193   |
|               |                |         |          | 29   | 51   | 83   | 118  | 151   | 166   |
| efficient     | classification | 7.3     | 7.1      | 61   | 99   | 172  | 320  | 637   | 1245  |
|               |                |         |          | 16   | 20   | 23   | 25   | 25    | 26    |

---

## format

1. model description (Just need to complete the table below)

| model name                                           | batch size | duration |
|------------------------------------------------------|------------|----------|
| faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8 | 1          | 233      |
|                                                      | 2          | 235      |
|                                                      | 4          | 236      |
|                                                      | 8          | 240      |
|                                                      | 16         | 249      |
|                                                      | 32         | 266      |
| ssd_resnet50_v1_fpn_640x640_coco17_tpu-8             | 1          | 34.4     |
|                                                      | 2          | 36.1     |
|                                                      | 4          | 38.1     |
|                                                      | 8          | 40.4     |
|                                                      | 16         | 49.9     |
|                                                      | 32         | 65.7     |

2. model files:

   * put the model files into [PROJECT_ROOT/data/model/inference](../../data/model/inference)
3. python script:

   * You need to write a simple script that uses tf-serving for model inference and can specify different models, batch sizes.
   * put the script into [PROJECT_ROOT/test/inference](../../test/inference)
4. profiling files

   * For each model (e.g. **Fast-RCNN-xxx**), you need to create a directory in [PROJECT_ROOT/workload/profiling](./)
   * Put these files into the directory:

     * Fast-RCNN-xxx.csv

       ```
       batch_size, duration
       1, 25.9
       2, 32.8
       ..., ...
       32, 182.1
       ```
     * Fast-RCNN-xxx-b1.csv

       ```
       24.5
       25.6
       26.1
       ...
       25.8  (1000 lines in total)
       ```
     * Fast-RCNN-xxx-b2.csv
     * ......
     * Fast-RCNN-xxx-b32.csv
