import { useEffect } from "react";
import { useTranslation } from "react-i18next";

const LanguageSwitcher = () => {
  const { i18n, t } = useTranslation();

  // EN: Change direction when language changes
  // FA: هنگام تغییر زبان، جهت متن تغییر می‌کند
  useEffect(() => {
    document.documentElement.dir = i18n.language === "fa" ? "rtl" : "ltr";
  }, [i18n.language]);

  const switchLang = (lng: "en" | "fa") => {
    i18n.changeLanguage(lng);
  };

  return (
    <div className="lang-switcher">
      <span>{t("language")}:</span>
      <button
        className={i18n.language === "en" ? "active" : ""}
        onClick={() => switchLang("en")}
      >
        {t("english")}
      </button>
      <button
        className={i18n.language === "fa" ? "active" : ""}
        onClick={() => switchLang("fa")}
      >
        {t("persian")}
      </button>
    </div>
  );
};

export default LanguageSwitcher;
