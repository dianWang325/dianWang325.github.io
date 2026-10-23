import type {
    SiteConfig,
    ProfileConfig,
    LicenseConfig,
} from "./types/config"
import type { FriendLink } from "./types/friend"
import type { I18nConfig } from "./types/i18n"

export const siteConfig: SiteConfig = {
    title: "Dian's Notes", // Title of the site, used in the tab in the browser and in SEO
    subTitle: "记录技术、阅读与持续思考", // Subtitle of the site
    rootSiteUrl: "https://dianwang325.github.io", // Root URL of the site, used for generating absolute URLs for SEO and social sharing

    favicon: "/favicon/favicon.ico", // Path of the favicon, relative to the /public directory

    pageSize: 6, // Number of posts per page
    toc: {
        enable: true,
        depth: 3 // Max depth of the table of contents, between 1 and 4
    },
    blogNavi: {
        enable: true // Whether to enable blog navigation in the blog footer
    },
    comments: {
        enable: false, // Comments can be enabled after a backend is configured
        platform: "default", // Comment platform, set "default" to use Momo-backend, also supports "twikoo"
        backendUrl: "" // Backend URL for comments
    },
    theme: {
        AOS: true, // Whether to enable AOS (Animate On Scroll) for animations
        LQIP: true, // Whether to enable LQIP (Low-Quality Image Placeholder) for image placeholders
        PhotoSwipe: true, // Whether to enable PhotoSwipe for image viewer
        imageCollage: {
            enable: true, // Whether to automatically arrange consecutive images into a grid (collage)
            maxColumns: 4 // Max images per row in a collage (2 - 6); the actual number is chosen automatically
        },
        postCard: {
            imageMode: "top" // Cover image mode for article cards: "top" shows the image above the content; "background" uses the image as the card background, fading to transparent from right to left
        }
    },
    expressiveCode: {
        enable: true, // Whether to enable Expressive Code for code blocks; when false, code blocks fall back to plain text without highlighting (same in the CMS preview)
        theme: "one-dark-pro" // Shiki theme of code blocks, e.g. "one-dark-pro", "github-dark", "vitesse-dark"; one theme is used for both light and dark mode
    }
}

export const profileConfig: ProfileConfig = {
    avatar: "https://github.com/dianWang325.png", // GitHub avatar
    name: "Dian", // Used in the footer of the blog
    description: "记录技术、阅读与持续思考。", // Used in SEO
    indexPage: "https://github.com/dianWang325", // The homepage, used in footer and SEO
    startYear: 2026, // The year the site was created, used in the footer
    links: [
        {
            name: "GitHub",
            url: "https://github.com/dianWang325",
            icon: "simple-icons:github",
            color: "#181717"
        }
    ]
}

export const licenseConfig: LicenseConfig = {
	enable: true, // Whether to enable license information
	name: "CC BY-NC-SA 4.0", // License name
	url: "https://creativecommons.org/licenses/by-nc-sa/4.0/", // License URL
};

export const i18nConfig: I18nConfig = {
    defaultLanguage: "zh-cn", // Default language of the site
    supportedLanguages: ["zh-cn", "en"], // List of supported languages
    translations: { // Translation content for each supported language
        "zh-cn": {
            Cover: {
                title: {
                    home: "欢迎来到 Dian's Notes",
                    archive: "文章归档",
                    about: "关于",
                    friends: "链接",
                },
                subTitle: {
                    home: "记录技术、阅读与持续思考",
                    archive: "共 {count} 篇文章", // {count} will be replaced with the total number of articles
                    about: "关于这个博客与它的作者",
                    friends: "值得关注的站点",
                }
            }
        },
        "en": {
            Cover: {
                title: {
                    home: "Welcome to Dian's Notes",
                    archive: "Archive",
                    about: "About",
                    friends: "Friends",
                },
                subTitle: {
                    home: "Notes on technology, reading, and continuous learning",
                    archive: "Total of {count} articles",
                    about: "About this blog and its author",
                    friends: "Sites worth following",
                }
            }
        }
    }
};

export const friendLinkConfig: FriendLink[] = [
    {
        name: 'Dian on GitHub',
        avatar: 'https://github.com/dianWang325.png',
        url: 'https://github.com/dianWang325',
        description: 'Projects, experiments, and notes.'
    }
    // Add more friend links here
]
