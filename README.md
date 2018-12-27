# CoreML-MPS

## Introduction

Inspired by: [A peek inside Core ML](http://machinethink.net/blog/peek-inside-coreml/)

This is a proof-of-concept project. It directly reads weights and bias from compiled CoreML v1 model and use them in MPSNN.

The demo is an implementation of [waifu2x-ios](https://github.com/imxieyi/waifu2x-ios).

## Warning

This is generally a hack and is not documented by Apple in any forms. **DO NOT** try to use this in production environment since you will never know when Apple will change the compiled mlmodel format.

## Screenshot

![demo](/screenshot.jpeg)

Image source: [https://www.pixiv.net/member_illust.php?mode=medium&illust_id=45068168](https://www.pixiv.net/member_illust.php?mode=medium&illust_id=45068168) 
