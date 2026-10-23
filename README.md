# Dian's Notes

Dian 的个人知识博客，基于 [Momo](https://github.com/Motues/Momo) 与 Astro 构建，部署目标为 GitHub Pages。

## 本地运行

需要 Node.js 22 或更高版本，以及 pnpm。

```bash
pnpm install
pnpm dev
```

本地编辑器默认地址为 `http://localhost:4321`。

## 写一篇文章

```bash
pnpm momo new notes/my-first-post
```

也可以在 `src/content/blog/` 中创建文章目录，并添加 `zh-cn.md` 和可选的 `en.md`。

## 修改博客信息

网站名称、简介、作者信息、语言和主题开关位于 `src/config.ts`。

## 发布

将项目推送到 `dianWang325/dianWang325.github.io` 的 `main` 分支后，仓库内的 GitHub Actions 工作流会自动构建并发布网站。

首次发布前，需要在 GitHub 仓库的 **Settings → Pages → Build and deployment** 中选择 **GitHub Actions**。

## 许可

主题代码沿用原项目的 MIT License；博客文章版权归 Dian 所有，文章页面默认标注 CC BY-NC-SA 4.0。
