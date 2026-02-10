"use client";

import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ErrorBoundary } from "./components/ErrorBoundary";

interface ResearchResponse {
  query: string;
  report: string;
  sources: string[];
  confidence: number;
  iteration_count: number;
  quality_score?: number;
  error?: string;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Timer using requestAnimationFrame (prevents throttling when tab is backgrounded)
  useEffect(() => {
    if (!isLoading || startTime === null) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const updateTimer = () => {
      const now = Date.now();
      const elapsed = (now - startTime) / 1000; // Convert to seconds
      setElapsedTime(elapsed);
      animationFrameRef.current = requestAnimationFrame(updateTimer);
    };

    animationFrameRef.current = requestAnimationFrame(updateTimer);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isLoading, startTime]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);
    setCurrentStep(null);
    setElapsedTime(0);
    setStartTime(Date.now());

    try {
      const response = await fetch("/api/research", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Research request failed");
      }

      // Handle NDJSON streaming
      if (!response.body) {
        throw new Error("Response body is null");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          // Process any remaining buffer
          if (buffer.trim()) {
            const lines = buffer.split("\n").filter((line) => line.trim());
            for (const line of lines) {
              try {
                const data = JSON.parse(line);
                if (data.type === "result") {
                  setResult(data.report);
                  if (startTime) {
                    const finalDuration = (Date.now() - startTime) / 1000;
                    setElapsedTime(finalDuration);
                  }
                } else if (data.type === "error") {
                  throw new Error(data.error);
                }
              } catch (parseErr) {
                // Skip invalid JSON lines
                console.warn("Failed to parse line:", line, parseErr);
              }
            }
          }
          break;
        }

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete lines (ending with \n)
        const lines = buffer.split("\n");
        // Keep the last incomplete line in buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const data = JSON.parse(line);

            if (data.type === "log") {
              // Update current step status
              setCurrentStep(data.node || data.content);
            } else if (data.type === "result") {
              // Set final result and stop loading
              setResult(data.report);
              if (startTime) {
                const finalDuration = (Date.now() - startTime) / 1000;
                setElapsedTime(finalDuration);
              }
              setIsLoading(false);
            } else if (data.type === "error") {
              throw new Error(data.error);
            }
          } catch (parseErr) {
            // Skip invalid JSON lines
            console.warn("Failed to parse line:", line, parseErr);
          }
        }
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
      setIsLoading(false);
    } finally {
      setStartTime(null);
    }
  };

  const handleReset = () => {
    setQuery("");
    setResult(null);
    setError(null);
    setCurrentStep(null);
    setElapsedTime(0);
    setIsLoading(false);
    setStartTime(null);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  return (
    <div className="min-h-screen bg-black text-green-400 font-mono p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl mb-8 text-green-500">DEEP RESEARCH CONSOLE</h1>

        {!result && !isLoading && (
          <form onSubmit={handleSubmit} className="mb-8">
            <div className="flex gap-4">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter research query..."
                className="flex-1 bg-black border border-green-500 px-4 py-2 text-green-400 placeholder-green-600 focus:outline-none focus:border-green-400"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !query.trim()}
                className="px-6 py-2 bg-green-500 text-black font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:bg-green-400 transition-colors"
              >
                RESEARCH
              </button>
            </div>
          </form>
        )}

        {/* Stasis State: Waiting */}
        {isLoading && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-green-400">T+ {elapsedTime.toFixed(1)}s</span>
              <span className="animate-pulse text-green-500">_</span>
            </div>
            <div className="text-green-600 text-sm">
              {currentStep ? (
                <>Step: <span className="text-green-400">{currentStep}</span></>
              ) : (
                "Executing research agent... This may take several minutes."
              )}
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="p-4 border border-red-500 text-red-400">
            <div className="font-semibold mb-2">ERROR:</div>
            <div>{error}</div>
            <button
              onClick={handleReset}
              className="mt-4 px-4 py-2 bg-red-500 text-black hover:bg-red-400 transition-colors"
            >
              Reset
            </button>
          </div>
        )}

        {/* Generative State: Results Arrival */}
        {result && !isLoading && (
          <ErrorBoundary>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="space-y-6"
            >
              {/* Metadata */}
              <div className="border-b border-green-500 pb-4">
                <div className="text-green-500 font-semibold mb-2">
                  STATUS: COMPLETE
                </div>
                <div className="text-sm text-green-600 space-y-1">
                  <div>Duration: {elapsedTime.toFixed(1)}s</div>
                  <div>Confidence: {(result.confidence * 100).toFixed(1)}%</div>
                  {result.quality_score !== undefined && (
                    <div>Quality Score: {(result.quality_score * 100).toFixed(1)}%</div>
                  )}
                  <div>Iterations: {result.iteration_count}</div>
                </div>
              </div>

              {/* Report Content */}
              <div className="prose prose-invert prose-green max-w-none text-green-400 [&>*]:mb-4 [&>h1]:text-green-300 [&>h1]:text-2xl [&>h1]:font-bold [&>h2]:text-green-300 [&>h2]:text-xl [&>h2]:font-semibold [&>h3]:text-green-300 [&>h3]:text-lg [&>p]:leading-relaxed [&>ul]:list-disc [&>ul]:ml-6 [&>ol]:list-decimal [&>ol]:ml-6 [&>code]:bg-green-900 [&>code]:px-1 [&>code]:rounded [&>pre]:bg-green-900 [&>pre]:p-4 [&>pre]:rounded [&>pre]:overflow-x-auto [&>blockquote]:border-l-4 [&>blockquote]:border-green-500 [&>blockquote]:pl-4 [&>blockquote]:italic">
                <ReactMarkdown>
                  {result.report}
                </ReactMarkdown>
              </div>

              {/* Sources */}
              {result.sources && result.sources.length > 0 && (
                <div className="border-t border-green-500 pt-4">
                  <div className="text-green-500 font-semibold mb-2">SOURCES:</div>
                  <ul className="list-disc list-inside space-y-1 text-sm text-green-600">
                    {result.sources.map((source, idx) => (
                      <li key={idx}>
                        <a
                          href={source}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:text-green-400 underline"
                        >
                          {source}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Reset Button */}
              <button
                onClick={handleReset}
                className="px-6 py-2 bg-green-500 text-black font-semibold hover:bg-green-400 transition-colors"
              >
                New Query
              </button>
            </motion.div>
          </ErrorBoundary>
        )}
      </div>
    </div>
  );
}
