import { PredictPanel } from "@/components/predict-panel";

export default function Home() {
  return (
    <div className="flex min-h-full flex-1 flex-col bg-zinc-950 px-4 py-12 sm:px-8">
      <PredictPanel />
    </div>
  );
}
