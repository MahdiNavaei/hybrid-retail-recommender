import { useTranslation } from "react-i18next";

interface Props {
  online: boolean;
}

const StatusChip = ({ online }: Props) => {
  const { t } = useTranslation();
  return (
    <div className={`status-chip ${online ? "online" : "offline"}`}>
      <span className="dot" />
      <span>{online ? t("backendOnline") : t("backendOffline")}</span>
    </div>
  );
};

export default StatusChip;
