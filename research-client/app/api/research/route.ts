import { NextRequest, NextResponse } from "next/server";

// CRITICAL: Force the Edge Runtime
// (Higher concurrency, better for streaming, longer timeouts)
export const runtime = "edge";

const BACKEND_URL = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "";

export async function POST(request: NextRequest) {
  if (!BACKEND_URL) {
    return NextResponse.json(
      { error: "Server Configuration Error: API_URL is missing." },
      { status: 500 }
    );
  }

  try {
    const body = await request.json();
    const { query } = body;

    // Construct the backend URL
    // Remove trailing slash if present to avoid double slash
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

    // CRITICAL FIX: Return the stream directly!
    // Do NOT use 'await response.json()'
    return new NextResponse(backendResponse.body, {
      status: 200,
      headers: {
        // Ensure browser treats this as a stream
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