import { NextRequest, NextResponse } from "next/server";

// CRITICAL: Force the Edge Runtime for streaming support
export const runtime = "edge";

// FIX: Use the correct environment variable from Vercel
const BACKEND_URL = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "";

export async function POST(request: NextRequest) {
  // Debug log (visible in Vercel functions tab)
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

    // Clean the URL to avoid double slashes
    const cleanUrl = BACKEND_URL.replace(/\/$/, "");
    const targetUrl = `${cleanUrl}/research`;

    console.log(`Streaming request to: ${targetUrl}`);

    // Forward the request to Cloud Run
    const backendResponse = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      return NextResponse.json(
        { error: `Backend Error ${backendResponse.status}: ${errorText}` },
        { status: backendResponse.status }
      );
    }

    // CRITICAL: Return the stream directly (No 'await json()')
    return new NextResponse(backendResponse.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });

  } catch (error: any) {
    console.error("Proxy Error:", error);
    return NextResponse.json(
      { error: `Proxy Error: ${error.message}` },
      { status: 500 }
    );
  }
}