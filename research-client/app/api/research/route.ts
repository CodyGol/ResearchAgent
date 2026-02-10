import { NextRequest, NextResponse } from "next/server";

// FIX: Use the variable names we actually set in Vercel
const BACKEND_URL = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "";
const TIMEOUT_MS = 600000; // 10 minutes

export async function POST(request: NextRequest) {
  // Debug log to see what the server actually sees (check Vercel logs if this fails)
  console.log("Connecting to Backend:", BACKEND_URL);

  if (!BACKEND_URL) {
    return NextResponse.json(
      { error: "Server Configuration Error: API_URL is missing." },
      { status: 500 }
    );
  }

  try {
    const body = await request.json();
    const { query } = body;

    if (!query) {
      return NextResponse.json(
        { error: "Query is required" },
        { status: 400 }
      );
    }

    // Create AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      // Ensure we don't have double slashes if the URL ends with /
      const cleanUrl = BACKEND_URL.replace(/\/$/, "");
      const targetUrl = `${cleanUrl}/research`; 

      console.log(`Forwarding request to: ${targetUrl}`);

      const response = await fetch(targetUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        // Try to parse the error from the backend
        const errorText = await response.text();
        console.error("Backend Error:", response.status, errorText);
        return NextResponse.json(
          { error: `Backend Error ${response.status}: ${errorText}` },
          { status: response.status }
        );
      }

      const data = await response.json();
      return NextResponse.json(data);

    } catch (fetchError: any) {
      clearTimeout(timeoutId);
      console.error("Fetch failed:", fetchError);
      
      if (fetchError.name === "AbortError") {
        return NextResponse.json({ error: "Request timed out (10m limit)" }, { status: 504 });
      }
      return NextResponse.json({ error: `Connection Failed: ${fetchError.message}` }, { status: 502 });
    }

  } catch (error: any) {
    return NextResponse.json({ error: `Invalid Request: ${error.message}` }, { status: 400 });
  }
}