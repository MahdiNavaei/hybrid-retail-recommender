import { ReactNode } from "react";
import Header from "./Header";
import "./layout.css";

interface Props {
  children: ReactNode;
  backendOnline: boolean;
}

const AppLayout = ({ children, backendOnline }: Props) => {
  return (
    <div className="app-shell">
      <Header backendOnline={backendOnline} />
      <main className="app-main">{children}</main>
    </div>
  );
};

export default AppLayout;
