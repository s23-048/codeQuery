// Test TypeScript file

interface Config {
    apiUrl: string;
    timeout: number;
}

function createClient(config: Config): object {
    return { url: config.apiUrl };
}

const handleRequest = async (req: Request): Promise<Response> => {
    const data = await req.json();
    return new Response(JSON.stringify(data));
};

class ApiService {
    private config: Config;

    constructor(config: Config) {
        this.config = config;
    }

    async fetchAll(): Promise<any[]> {
        const response = await fetch(this.config.apiUrl);
        return response.json();
    }
}

export { ApiService, createClient };
