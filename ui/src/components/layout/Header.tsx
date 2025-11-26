import { useTranslation } from "react-i18next";
import LanguageSwitcher from "./LanguageSwitcher";
import StatusChip from "./StatusChip";

interface Props {
  backendOnline: boolean;
}

const Header = ({ backendOnline }: Props) => {
  const { t } = useTranslation();
  return (
    <header className="app-header">
      <div>
        <h1 className="app-title">{t("appTitle")}</h1>
        <p className="app-subtitle">{t("subtitle")}</p>
      </div>
      <div className="header-actions">
        <StatusChip online={backendOnline} />
        <LanguageSwitcher />
      </div>
    </header>
  );
};

export default Header;
