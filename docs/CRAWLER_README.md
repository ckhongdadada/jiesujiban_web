# 词典扩充爬虫说明

本文件对应新增脚本：
- [beijing_alias_crawler.py](C:/Users/28414/PycharmProjects/接诉即办项目/crawlers/beijing_alias_crawler.py)

## 目标

在不修改原始词典文件的前提下，把新抓到的小区名、站点名、街道名写入：
- [beijing_districts_extra.json](C:/Users/28414/PycharmProjects/接诉即办项目/data/beijing_districts_extra.json)

增强版解析器 [location_ner.py](C:/Users/28414/PycharmProjects/接诉即办项目/enhancements/location_ner.py) 已经会自动加载这个扩展词典。

## 当前支持的来源

- `lianjia`
- `anjuke`
- `subway`
- `amap_subway`
- `amap_streets`
- `amap_communities`

其中：
- `lianjia` 和 `anjuke` 目前都可能遇到登录/反爬页面，脚本会把这个情况记入 `crawl_report.json`
- `amap_subway` 和 `amap_streets` 需要先设置环境变量 `AMAP_API_KEY`

## 运行方式

只看抓取结果，不写文件：

```powershell
& C:\Users\28414\anaconda3\envs\qwen_env\python.exe C:\Users\28414\PycharmProjects\接诉即办项目\crawlers\beijing_alias_crawler.py --dry-run
```

写入扩展词典：

```powershell
& C:\Users\28414\anaconda3\envs\qwen_env\python.exe C:\Users\28414\PycharmProjects\接诉即办项目\crawlers\beijing_alias_crawler.py --max-pages 3
```

使用高德 API 强化站点和街道抓取：

```powershell
$env:AMAP_API_KEY='你的key'
& C:\Users\28414\anaconda3\envs\qwen_env\python.exe C:\Users\28414\PycharmProjects\接诉即办项目\crawlers\beijing_alias_crawler.py --sources lianjia anjuke amap_subway amap_streets --max-pages 2
```

## 输出文件

- `data/beijing_districts_extra.json`
- `data/beijing_place_records.jsonl`
- `data/crawl_report.json`
